import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
class TestCudaMemoryFunctions(ContextResettingTestCase):

    def setUp(self):
        super().setUp()
        self.context = devices.get_context()

    def tearDown(self):
        del self.context
        super(TestCudaMemoryFunctions, self).tearDown()

    def test_memcpy(self):
        hstary = np.arange(100, dtype=np.uint32)
        hstary2 = np.arange(100, dtype=np.uint32)
        sz = hstary.size * hstary.dtype.itemsize
        devary = self.context.memalloc(sz)
        driver.host_to_device(devary, hstary, sz)
        driver.device_to_host(hstary2, devary, sz)
        self.assertTrue(np.all(hstary == hstary2))

    def test_memset(self):
        dtype = np.dtype('uint32')
        n = 10
        sz = dtype.itemsize * 10
        devary = self.context.memalloc(sz)
        driver.device_memset(devary, 171, sz)
        hstary = np.empty(n, dtype=dtype)
        driver.device_to_host(hstary, devary, sz)
        hstary2 = np.array([2880154539] * n, dtype=np.dtype('uint32'))
        self.assertTrue(np.all(hstary == hstary2))

    def test_d2d(self):
        hst = np.arange(100, dtype=np.uint32)
        hst2 = np.empty_like(hst)
        sz = hst.size * hst.dtype.itemsize
        dev1 = self.context.memalloc(sz)
        dev2 = self.context.memalloc(sz)
        driver.host_to_device(dev1, hst, sz)
        driver.device_to_device(dev2, dev1, sz)
        driver.device_to_host(hst2, dev2, sz)
        self.assertTrue(np.all(hst == hst2))