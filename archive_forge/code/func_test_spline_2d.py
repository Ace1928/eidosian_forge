import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_spline_2d(self):
    x, y, z = self._sample_2d_data()
    lut = RectBivariateSpline(x, y, z)
    xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
    assert_array_almost_equal(interpn((x, y), z, xi, method='splinef2d'), lut.ev(xi[:, 0], xi[:, 1]))