import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_nvvm_bad_option(self):
    msg = '-made-up-option=2 is an unsupported option'
    with self.assertRaisesRegex(NvvmError, msg):
        nvvm.llvm_to_ptx('', made_up_option=2)