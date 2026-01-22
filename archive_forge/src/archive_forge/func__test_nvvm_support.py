import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def _test_nvvm_support(self, arch):
    compute_xx = 'compute_{0}{1}'.format(*arch)
    nvvmir = self.get_nvvmir()
    ptx = nvvm.llvm_to_ptx(nvvmir, arch=compute_xx, ftz=1, prec_sqrt=0, prec_div=0).decode('utf8')
    self.assertIn('.target sm_{0}{1}'.format(*arch), ptx)
    self.assertIn('simple', ptx)
    self.assertIn('ave', ptx)