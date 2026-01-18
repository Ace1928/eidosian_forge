import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_out_of_bounds_extrap2(self):
    points, values = self._get_sample_4d_2()
    interp = RegularGridInterpolator(points, values, bounds_error=False, fill_value=None)
    sample = np.asarray([[-0.1, -0.1, -0.1, -0.1], [1.1, 1.1, 1.1, 1.1], [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
    wanted = np.asarray([0.0, 11.0, 11.0, 11.0])
    assert_array_almost_equal(interp(sample, method='nearest'), wanted)
    wanted = np.asarray([-12.1, 133.1, -1069.0, -97.9])
    assert_array_almost_equal(interp(sample, method='linear'), wanted)