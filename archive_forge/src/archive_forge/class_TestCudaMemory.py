import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestCudaMemory(ContextResettingTestCase):

    def setUp(self):
        super().setUp()
        self.context = devices.get_context()

    def tearDown(self):
        del self.context
        super(TestCudaMemory, self).tearDown()

    def _template(self, obj):
        self.assertTrue(driver.is_device_memory(obj))
        driver.require_device_memory(obj)
        if driver.USE_NV_BINDING:
            expected_class = driver.binding.CUdeviceptr
        else:
            expected_class = drvapi.cu_device_ptr
        self.assertTrue(isinstance(obj.device_ctypes_pointer, expected_class))

    def test_device_memory(self):
        devmem = self.context.memalloc(1024)
        self._template(devmem)

    def test_device_view(self):
        devmem = self.context.memalloc(1024)
        self._template(devmem.view(10))

    def test_host_alloc(self):
        devmem = self.context.memhostalloc(1024, mapped=True)
        self._template(devmem)

    def test_pinned_memory(self):
        ary = np.arange(10)
        devmem = self.context.mempin(ary, ary.ctypes.data, ary.size * ary.dtype.itemsize, mapped=True)
        self._template(devmem)

    def test_managed_memory(self):
        devmem = self.context.memallocmanaged(1024)
        self._template(devmem)

    def test_derived_pointer(self):

        def handle_val(mem):
            if driver.USE_NV_BINDING:
                return int(mem.handle)
            else:
                return mem.handle.value

        def check(m, offset):
            v1 = m.view(offset)
            self.assertEqual(handle_val(v1.owner), handle_val(m))
            self.assertEqual(m.refct, 2)
            self.assertEqual(handle_val(v1) - offset, handle_val(v1.owner))
            v2 = v1.view(offset)
            self.assertEqual(handle_val(v2.owner), handle_val(m))
            self.assertEqual(handle_val(v2.owner), handle_val(m))
            self.assertEqual(handle_val(v2) - offset * 2, handle_val(v2.owner))
            self.assertEqual(m.refct, 3)
            del v2
            self.assertEqual(m.refct, 2)
            del v1
            self.assertEqual(m.refct, 1)
        m = self.context.memalloc(1024)
        check(m=m, offset=0)
        check(m=m, offset=1)

    def test_user_extension(self):
        fake_ptr = ctypes.c_void_p(3735928559)
        dtor_invoked = [0]

        def dtor():
            dtor_invoked[0] += 1
        ptr = driver.MemoryPointer(context=self.context, pointer=fake_ptr, size=40, finalizer=dtor)
        self.assertEqual(dtor_invoked[0], 0)
        del ptr
        self.assertEqual(dtor_invoked[0], 1)
        ptr = driver.MemoryPointer(context=self.context, pointer=fake_ptr, size=40, finalizer=dtor)
        owned = ptr.own()
        del owned
        self.assertEqual(dtor_invoked[0], 1)
        del ptr
        self.assertEqual(dtor_invoked[0], 2)