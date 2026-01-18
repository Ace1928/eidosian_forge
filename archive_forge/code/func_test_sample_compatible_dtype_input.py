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
@pytest.mark.parametrize('typ', [x for x in np.typecodes['All'][:20] if x not in 'gG'])
def test_sample_compatible_dtype_input(self, typ):
    n = 4
    a = self.rng.random([n, n]).astype(typ)
    assert isinstance(det(a), (np.float64, np.complex128))