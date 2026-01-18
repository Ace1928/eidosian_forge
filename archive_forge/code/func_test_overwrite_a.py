import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_overwrite_a(self):
    a = np.arange(9).reshape(3, 3).astype(np.float32)
    ac = a.copy()
    deta = det(ac, overwrite_a=True)
    assert_allclose(deta, 0.0)
    assert not (a == ac).all()