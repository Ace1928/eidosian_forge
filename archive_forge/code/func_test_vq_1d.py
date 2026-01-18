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
def test_vq_1d(self, xp):
    data = X[:, 0]
    initc = data[:3]
    a, b = _vq.vq(data, initc)
    data = xp.asarray(data)
    initc = xp.asarray(initc)
    ta, tb = py_vq(data[:, np.newaxis], initc[:, np.newaxis])
    xp_assert_equal(ta, xp.asarray(a, dtype=xp.int64), check_dtype=False)
    xp_assert_equal(tb, xp.asarray(b))