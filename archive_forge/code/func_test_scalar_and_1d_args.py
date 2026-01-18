import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def test_scalar_and_1d_args(self):
    a = block_diag(1)
    assert_equal(a.shape, (1, 1))
    assert_array_equal(a, [[1]])
    a = block_diag([2, 3], 4)
    assert_array_equal(a, [[2, 3, 0], [0, 0, 4]])