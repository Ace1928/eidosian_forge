import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
@skip_on_cudasim('CUDA HW required')
class Test3rdPartyContext(CUDATestCase):

    def tearDown(self):
        super().tearDown()
        cuda.close()

    def test_attached_primary(self, extra_work=lambda: None):
        the_driver = driver.driver
        if driver.USE_NV_BINDING:
            dev = driver.binding.CUdevice(0)
            hctx = the_driver.cuDevicePrimaryCtxRetain(dev)
        else:
            dev = 0
            hctx = driver.drvapi.cu_context()
            the_driver.cuDevicePrimaryCtxRetain(byref(hctx), dev)
        try:
            ctx = driver.Context(weakref.proxy(self), hctx)
            ctx.push()
            my_ctx = cuda.current_context()
            if driver.USE_NV_BINDING:
                self.assertEqual(int(my_ctx.handle), int(ctx.handle))
            else:
                self.assertEqual(my_ctx.handle.value, ctx.handle.value)
            extra_work()
        finally:
            ctx.pop()
            the_driver.cuDevicePrimaryCtxRelease(dev)

    def test_attached_non_primary(self):
        the_driver = driver.driver
        if driver.USE_NV_BINDING:
            flags = 0
            dev = driver.binding.CUdevice(0)
            hctx = the_driver.cuCtxCreate(flags, dev)
        else:
            hctx = driver.drvapi.cu_context()
            the_driver.cuCtxCreate(byref(hctx), 0, 0)
        try:
            cuda.current_context()
        except RuntimeError as e:
            self.assertIn('Numba cannot operate on non-primary CUDA context ', str(e))
        else:
            self.fail('No RuntimeError raised')
        finally:
            the_driver.cuCtxDestroy(hctx)

    def test_cudajit_in_attached_primary_context(self):

        def do():
            from numba import cuda

            @cuda.jit
            def foo(a):
                for i in range(a.size):
                    a[i] = i
            a = cuda.device_array(10)
            foo[1, 1](a)
            self.assertEqual(list(a.copy_to_host()), list(range(10)))
        self.test_attached_primary(do)