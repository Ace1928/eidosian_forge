import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
class TestMemoryView(MemoryLeakMixin, TestCase):
    """
    Test memoryview-specific attributes and operations.
    """

    def _arrays(self):
        arr = np.arange(12)
        yield arr
        arr = arr.reshape((3, 4))
        yield arr
        yield arr.T
        yield arr[::2]
        arr.setflags(write=False)
        yield arr
        arr = np.zeros(())
        assert arr.ndim == 0
        yield arr

    def test_ndim(self):
        for arr in self._arrays():
            m = memoryview(arr)
            self.assertPreciseEqual(ndim_usecase(m), arr.ndim)

    def test_shape(self):
        for arr in self._arrays():
            m = memoryview(arr)
            self.assertPreciseEqual(shape_usecase(m), arr.shape)

    def test_strides(self):
        for arr in self._arrays():
            m = memoryview(arr)
            self.assertPreciseEqual(strides_usecase(m), arr.strides)

    def test_itemsize(self):
        for arr in self._arrays():
            m = memoryview(arr)
            self.assertPreciseEqual(itemsize_usecase(m), arr.itemsize)

    def test_nbytes(self):
        for arr in self._arrays():
            m = memoryview(arr)
            self.assertPreciseEqual(nbytes_usecase(m), arr.size * arr.itemsize)

    def test_readonly(self):
        for arr in self._arrays():
            m = memoryview(arr)
            self.assertIs(readonly_usecase(m), not arr.flags.writeable)
        m = memoryview(b'xyz')
        self.assertIs(readonly_usecase(m), True)
        m = memoryview(bytearray(b'xyz'))
        self.assertIs(readonly_usecase(m), False)

    def test_contiguous(self):
        m = memoryview(bytearray(b'xyz'))
        self.assertIs(contiguous_usecase(m), True)
        self.assertIs(c_contiguous_usecase(m), True)
        self.assertIs(f_contiguous_usecase(m), True)
        for arr in self._arrays():
            m = memoryview(arr)
            self.assertIs(contiguous_usecase(m), arr.flags.f_contiguous or arr.flags.c_contiguous)
            self.assertIs(c_contiguous_usecase(m), arr.flags.c_contiguous)
            self.assertIs(f_contiguous_usecase(m), arr.flags.f_contiguous)