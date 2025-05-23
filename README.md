# OpenCL + OpenMP Adder in C++

This project demonstrates the use of **OpenCL** for GPU parallel processing and **OpenMP** for CPU parallelization to add two arrays of integers.

## 🚀 Features

- Uses OpenCL to add large arrays in parallel on GPU
- Uses OpenMP to perform the same addition on CPU
- Compares performance between GPU and CPU execution
- Checks result consistency

## 📁 File Structure

```
opencl_openmp_adder/
├── opencl_openmp_adder.cpp  # Main C++ source file using OpenCL + OpenMP
└── README.md                # Project documentation
```

## 🧪 How to Compile and Run

### Requirements

- GCC or Clang with OpenMP support
- OpenCL SDK (e.g., Intel, NVIDIA, AMD)
- C++17 compiler

### Compile

```bash
g++ -std=c++17 -fopenmp opencl_openmp_adder.cpp -lOpenCL -o adder
```

### Run

```bash
./adder
```

## ⚠️ Notes

- You must have a working OpenCL installation (driver and headers).
- Execution time and result validation are printed to stdout.
