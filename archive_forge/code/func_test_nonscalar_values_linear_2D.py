import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_nonscalar_values_linear_2D(self):
    method = 'linear'
    points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5), (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    rng = np.random.default_rng(1234)
    trailing_points = (3, 4)
    values = rng.random((6, 7, *trailing_points))
    sample = rng.random(2)
    interp = RegularGridInterpolator(points, values, method=method, bounds_error=False)
    v = interp(sample)
    assert v.shape == (1, *trailing_points)
    vs = np.empty(values.shape[-2:])
    for i in range(values.shape[-2]):
        for j in range(values.shape[-1]):
            interp = RegularGridInterpolator(points, values[..., i, j], method=method, bounds_error=False)
            vs[i, j] = interp(sample).item()
    v2 = np.expand_dims(vs, axis=0)
    assert_allclose(v, v2, atol=1e-14, err_msg=method)