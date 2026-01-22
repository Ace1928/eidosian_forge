from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
@skip_on_cudasim('Compilation unsupported in the simulator')
class TestCompileToPTXForCurrentDevice(CUDATestCase):

    def test_compile_ptx_for_current_device(self):

        def add(x, y):
            return x + y
        args = (float32, float32)
        ptx, resty = compile_ptx_for_current_device(add, args, device=True)
        device_cc = cuda.get_current_device().compute_capability
        cc = cuda.cudadrv.nvvm.find_closest_arch(device_cc)
        target = f'.target sm_{cc[0]}{cc[1]}'
        self.assertIn(target, ptx)