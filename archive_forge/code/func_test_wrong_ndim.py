import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_wrong_ndim(self):
    x = np.random.randn(30, 3)
    y = np.random.randn(30)
    assert_raises(ValueError, interpnd.CloughTocher2DInterpolator, x, y)