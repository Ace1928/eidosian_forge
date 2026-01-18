import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_syr2k_wrong_c(self):
    f = getattr(fblas, 'dsyr2k', None)
    if f is not None:
        assert_raises(Exception, f, **{'a': self.a, 'b': self.b, 'alpha': 1.0, 'c': np.zeros((15, 8))})