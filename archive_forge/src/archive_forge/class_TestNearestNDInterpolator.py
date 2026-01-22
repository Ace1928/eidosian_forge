import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
class TestNearestNDInterpolator:

    def test_nearest_options(self):
        npts, nd = (4, 3)
        x = np.arange(npts * nd).reshape((npts, nd))
        y = np.arange(npts)
        nndi = NearestNDInterpolator(x, y)
        opts = {'balanced_tree': False, 'compact_nodes': False}
        nndi_o = NearestNDInterpolator(x, y, tree_options=opts)
        assert_allclose(nndi(x), nndi_o(x), atol=1e-14)

    def test_nearest_list_argument(self):
        nd = np.array([[0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 2]])
        d = nd[:, 3:]
        NI = NearestNDInterpolator((d[0], d[1]), d[2])
        assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])
        NI = NearestNDInterpolator((d[0], d[1]), list(d[2]))
        assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])

    def test_nearest_query_options(self):
        nd = np.array([[0, 0.5, 0, 1], [0, 0, 0.5, 1], [0, 1, 1, 2]])
        delta = 0.1
        query_points = ([0 + delta, 1 + delta], [0 + delta, 1 + delta])
        NI = NearestNDInterpolator((nd[0], nd[1]), nd[2])
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) - 1e-07
        assert_array_equal(NI(query_points, distance_upper_bound=distance_upper_bound), [np.nan, np.nan])
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) - 1e-07
        p = np.inf
        assert_array_equal(NI(query_points, distance_upper_bound=distance_upper_bound, p=p), [0, 2])
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) + 1e-07
        assert_array_equal(NI(query_points, distance_upper_bound=distance_upper_bound), [0, 2])

    def test_nearest_query_valid_inputs(self):
        nd = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 2]])
        NI = NearestNDInterpolator((nd[0], nd[1]), nd[2])
        with assert_raises(TypeError):
            NI([0.5, 0.5], query_options='not a dictionary')