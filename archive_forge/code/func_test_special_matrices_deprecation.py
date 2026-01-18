import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
@pytest.mark.parametrize('func', [tri, tril, triu])
def test_special_matrices_deprecation(func):
    with pytest.warns(DeprecationWarning, match="'tri'/'tril/'triu'"):
        func(np.array([[1]]))