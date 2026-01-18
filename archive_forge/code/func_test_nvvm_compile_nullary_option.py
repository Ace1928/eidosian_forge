import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_nvvm_compile_nullary_option(self):
    if runtime.get_version() < (11, 5):
        self.skipTest('-gen-lto unavailable in this toolkit version')
    nvvmir = self.get_nvvmir()
    ltoir = nvvm.llvm_to_ptx(nvvmir, opt=3, gen_lto=None, arch='compute_52')
    self.assertEqual(ltoir[:4], b'\xedCN\x7f')