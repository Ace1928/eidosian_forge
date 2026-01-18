import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_nvvm_compile_simple(self):
    nvvmir = self.get_nvvmir()
    ptx = nvvm.llvm_to_ptx(nvvmir).decode('utf8')
    self.assertTrue('simple' in ptx)
    self.assertTrue('ave' in ptx)