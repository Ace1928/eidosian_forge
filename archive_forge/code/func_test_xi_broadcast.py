import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@parametrize_rgi_interp_methods
def test_xi_broadcast(self, method):
    x, y, values = self._sample_2d_data()
    points = (x, y)
    xi = np.linspace(0, 1, 2)
    yi = np.linspace(0, 3, 3)
    sample = (xi[:, None], yi[None, :])
    v1 = interpn(points, values, sample, method=method, bounds_error=False)
    assert_equal(v1.shape, (2, 3))
    xx, yy = np.meshgrid(xi, yi)
    sample = np.c_[xx.T.ravel(), yy.T.ravel()]
    v2 = interpn(points, values, sample, method=method, bounds_error=False)
    assert_allclose(v1, v2.reshape(v1.shape))