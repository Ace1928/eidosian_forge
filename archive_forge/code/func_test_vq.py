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
@skip_if_array_api
def test_vq(self):
    initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
    for tp in [np.asarray, matrix]:
        label1, dist = _vq.vq(tp(X), tp(initc))
        assert_array_equal(label1, LABEL1)
        tlabel1, tdist = vq(tp(X), tp(initc))