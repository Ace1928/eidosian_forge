import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
@skip_if_array_api_gpu
@array_api_compatible
def test_py_vq(self, xp):
    initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
    arrays = [xp.asarray] if SCIPY_ARRAY_API else [np.asarray, matrix]
    for tp in arrays:
        label1 = py_vq(tp(X), tp(initc))[0]
        xp_assert_equal(label1, xp.asarray(LABEL1, dtype=xp.int64), check_dtype=False)