from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
class TestZswap(BaseSwap):
    blas_func = fblas.zswap
    dtype = complex128