from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def as_aligned(arr, align, dtype, order='C'):
    aligned = aligned_array(arr.shape, align, dtype, order)
    aligned[:] = arr[:]
    return aligned