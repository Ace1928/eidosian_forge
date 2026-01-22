import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
class TestCirculant:

    def test_basic(self):
        y = circulant([1, 2, 3])
        assert_array_equal(y, [[1, 3, 2], [2, 1, 3], [3, 2, 1]])