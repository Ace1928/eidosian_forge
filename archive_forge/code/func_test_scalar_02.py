import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def test_scalar_02(self):
    c = array([1, 2, 3])
    t = toeplitz(c, array(1))
    assert_array_equal(t, [[1], [2], [3]])