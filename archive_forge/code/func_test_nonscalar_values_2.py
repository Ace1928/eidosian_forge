import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@parametrize_rgi_interp_methods
def test_nonscalar_values_2(self, method):
    points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5), (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0), (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]
    rng = np.random.default_rng(1234)
    trailing_points = (3, 2)
    values = rng.random((6, 7, 8, 9, *trailing_points))
    sample = rng.random(4)
    v = interpn(points, values, sample, method=method, bounds_error=False)
    assert v.shape == (1, *trailing_points)
    vs = [[interpn(points, values[..., i, j], sample, method=method, bounds_error=False) for i in range(values.shape[-2])] for j in range(values.shape[-1])]
    assert_allclose(v, np.asarray(vs).T, atol=1e-14, err_msg=method)