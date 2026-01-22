import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
class TestHadamard:

    def test_basic(self):
        y = hadamard(1)
        assert_array_equal(y, [[1]])
        y = hadamard(2, dtype=float)
        assert_array_equal(y, [[1.0, 1.0], [1.0, -1.0]])
        y = hadamard(4)
        assert_array_equal(y, [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
        assert_raises(ValueError, hadamard, 0)
        assert_raises(ValueError, hadamard, 5)