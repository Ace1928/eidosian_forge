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
@array_api_compatible
def test_whiten_not_finite(self, xp):
    arrays = [xp.asarray] if SCIPY_ARRAY_API else [np.asarray, matrix]
    for tp in arrays:
        for bad_value in (xp.nan, xp.inf, -xp.inf):
            obs = tp([[0.9874451, bad_value], [0.62093317, 0.19406729], [0.87545741, 0.00735733], [0.85124403, 0.26499712], [0.4506759, 0.45464607]])
            assert_raises(ValueError, whiten, obs)