import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy import ndimage
@pytest.mark.parametrize('order', [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize('mode', ['mirror', 'grid-wrap', 'reflect'])
def test_spline_filter_vs_matrix_solution(order, mode):
    n = 100
    eye = np.eye(n, dtype=float)
    spline_filter_axis_0 = ndimage.spline_filter1d(eye, axis=0, order=order, mode=mode)
    spline_filter_axis_1 = ndimage.spline_filter1d(eye, axis=1, order=order, mode=mode)
    matrix = make_spline_knot_matrix(n, order, mode=mode)
    assert_almost_equal(eye, np.dot(spline_filter_axis_0, matrix))
    assert_almost_equal(eye, np.dot(spline_filter_axis_1, matrix.T))