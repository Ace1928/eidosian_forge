import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_nvvm_support(self):
    """Test supported CC by NVVM
        """
    for arch in nvvm.get_supported_ccs():
        self._test_nvvm_support(arch=arch)