import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test_pdist_chebyshev_iris(self):
    eps = 1e-14
    X = eo['iris']
    Y_right = eo['pdist-chebyshev-iris']
    Y_test1 = pdist(X, 'chebyshev')
    assert_allclose(Y_test1, Y_right, rtol=eps)