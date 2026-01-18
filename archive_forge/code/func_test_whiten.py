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
def test_whiten(self, xp):
    desired = xp.asarray([[5.08738849, 2.97091878], [3.19909255, 0.6966058], [4.51041982, 0.02640918], [4.38567074, 0.95120889], [2.3219148, 1.63195503]])
    obs = xp.asarray([[0.9874451, 0.82766775], [0.62093317, 0.19406729], [0.87545741, 0.00735733], [0.85124403, 0.26499712], [0.4506759, 0.45464607]])
    xp_assert_close(whiten(obs), desired, rtol=1e-05)