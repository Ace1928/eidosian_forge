import sys
from numba import cuda, njit
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.cudapy.cache_usecases import CUDAUseCase, UseCase
class _TestModule(CUDATestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        self.assertPreciseEqual(mod.assign_cpu(5), 5)
        self.assertPreciseEqual(mod.assign_cpu(5.5), 5.5)
        self.assertPreciseEqual(mod.assign_cuda(5), 5)
        self.assertPreciseEqual(mod.assign_cuda(5.5), 5.5)