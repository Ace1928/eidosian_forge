import numpy as np
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature
from numba.cuda import libdevice
@skip_on_cudasim('Libdevice functions are not supported on cudasim')
class TestLibdevice(CUDATestCase):
    """
    Some tests of libdevice function wrappers that check the returned values.

    These are mainly to check that the generation of the implementations
    results in correct typing and lowering for each type of function return
    (e.g. scalar return, UniTuple return, Tuple return, etc.).
    """

    def test_sincos(self):
        arr = np.arange(100, dtype=np.float64)
        sres = np.zeros_like(arr)
        cres = np.zeros_like(arr)
        cufunc = cuda.jit(use_sincos)
        cufunc[4, 32](sres, cres, arr)
        np.testing.assert_allclose(np.cos(arr), cres)
        np.testing.assert_allclose(np.sin(arr), sres)

    def test_frexp(self):
        arr = np.linspace(start=1.0, stop=10.0, num=100, dtype=np.float64)
        fracres = np.zeros_like(arr)
        expres = np.zeros(shape=arr.shape, dtype=np.int32)
        cufunc = cuda.jit(use_frexp)
        cufunc[4, 32](fracres, expres, arr)
        frac_expect, exp_expect = np.frexp(arr)
        np.testing.assert_array_equal(frac_expect, fracres)
        np.testing.assert_array_equal(exp_expect, expres)

    def test_sad(self):
        x = np.arange(0, 200, 2)
        y = np.arange(50, 150)
        z = np.arange(15, 115)
        r = np.zeros_like(x)
        cufunc = cuda.jit(use_sad)
        cufunc[4, 32](r, x, y, z)
        np.testing.assert_array_equal(np.abs(x - y) + z, r)