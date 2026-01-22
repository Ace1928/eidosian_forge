import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
class TestHistogramdd:

    def test_simple(self):
        x = np.array([[-0.5, 0.5, 1.5], [-0.5, 1.5, 2.5], [-0.5, 2.5, 0.5], [0.5, 0.5, 1.5], [0.5, 1.5, 2.5], [0.5, 2.5, 2.5]])
        H, edges = histogramdd(x, (2, 3, 3), range=[[-1, 1], [0, 3], [0, 3]])
        answer = np.array([[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1], [0, 0, 1]]])
        assert_array_equal(H, answer)
        ed = [[-2, 0, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
        H, edges = histogramdd(x, bins=ed, density=True)
        assert_(np.all(H == answer / 12.0))
        H, edges = histogramdd(x, (2, 3, 4), range=[[-1, 1], [0, 3], [0, 4]], density=True)
        answer = np.array([[[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]])
        assert_array_almost_equal(H, answer / 6.0, 4)
        z = [np.squeeze(y) for y in np.split(x, 3, axis=1)]
        H, edges = histogramdd(z, bins=(4, 3, 2), range=[[-2, 2], [0, 3], [0, 2]])
        answer = np.array([[[0, 0], [0, 0], [0, 0]], [[0, 1], [0, 0], [1, 0]], [[0, 1], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]])
        assert_array_equal(H, answer)
        Z = np.zeros((5, 5, 5))
        Z[list(range(5)), list(range(5)), list(range(5))] = 1.0
        H, edges = histogramdd([np.arange(5), np.arange(5), np.arange(5)], 5)
        assert_array_equal(H, Z)

    def test_shape_3d(self):
        bins = ((5, 4, 6), (6, 4, 5), (5, 6, 4), (4, 6, 5), (6, 5, 4), (4, 5, 6))
        r = np.random.rand(10, 3)
        for b in bins:
            H, edges = histogramdd(r, b)
            assert_(H.shape == b)

    def test_shape_4d(self):
        bins = ((7, 4, 5, 6), (4, 5, 7, 6), (5, 6, 4, 7), (7, 6, 5, 4), (5, 7, 6, 4), (4, 6, 7, 5), (6, 5, 7, 4), (7, 5, 4, 6), (7, 4, 6, 5), (6, 4, 7, 5), (6, 7, 5, 4), (4, 6, 5, 7), (4, 7, 5, 6), (5, 4, 6, 7), (5, 7, 4, 6), (6, 7, 4, 5), (6, 5, 4, 7), (4, 7, 6, 5), (4, 5, 6, 7), (7, 6, 4, 5), (5, 4, 7, 6), (5, 6, 7, 4), (6, 4, 5, 7), (7, 5, 6, 4))
        r = np.random.rand(10, 4)
        for b in bins:
            H, edges = histogramdd(r, b)
            assert_(H.shape == b)

    def test_weights(self):
        v = np.random.rand(100, 2)
        hist, edges = histogramdd(v)
        n_hist, edges = histogramdd(v, density=True)
        w_hist, edges = histogramdd(v, weights=np.ones(100))
        assert_array_equal(w_hist, hist)
        w_hist, edges = histogramdd(v, weights=np.ones(100) * 2, density=True)
        assert_array_equal(w_hist, n_hist)
        w_hist, edges = histogramdd(v, weights=np.ones(100, int) * 2)
        assert_array_equal(w_hist, 2 * hist)

    def test_identical_samples(self):
        x = np.zeros((10, 2), int)
        hist, edges = histogramdd(x, bins=2)
        assert_array_equal(edges[0], np.array([-0.5, 0.0, 0.5]))

    def test_empty(self):
        a, b = histogramdd([[], []], bins=([0, 1], [0, 1]))
        assert_array_max_ulp(a, np.array([[0.0]]))
        a, b = np.histogramdd([[], [], []], bins=2)
        assert_array_max_ulp(a, np.zeros((2, 2, 2)))

    def test_bins_errors(self):
        x = np.arange(8).reshape(2, 4)
        assert_raises(ValueError, np.histogramdd, x, bins=[-1, 2, 4, 5])
        assert_raises(ValueError, np.histogramdd, x, bins=[1, 0.99, 1, 1])
        assert_raises(ValueError, np.histogramdd, x, bins=[1, 1, 1, [1, 2, 3, -3]])
        assert_(np.histogramdd(x, bins=[1, 1, 1, [1, 2, 3, 4]]))

    def test_inf_edges(self):
        with np.errstate(invalid='ignore'):
            x = np.arange(6).reshape(3, 2)
            expected = np.array([[1, 0], [0, 1], [0, 1]])
            h, e = np.histogramdd(x, bins=[3, [-np.inf, 2, 10]])
            assert_allclose(h, expected)
            h, e = np.histogramdd(x, bins=[3, np.array([-1, 2, np.inf])])
            assert_allclose(h, expected)
            h, e = np.histogramdd(x, bins=[3, [-np.inf, 3, np.inf]])
            assert_allclose(h, expected)

    def test_rightmost_binedge(self):
        x = [0.9999999995]
        bins = [[0.0, 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 1.0)
        x = [1.0]
        bins = [[0.0, 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 1.0)
        x = [1.0000000001]
        bins = [[0.0, 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 0.0)
        x = [1.0001]
        bins = [[0.0, 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 0.0)

    def test_finite_range(self):
        vals = np.random.random((100, 3))
        histogramdd(vals, range=[[0.0, 1.0], [0.25, 0.75], [0.25, 0.5]])
        assert_raises(ValueError, histogramdd, vals, range=[[0.0, 1.0], [0.25, 0.75], [0.25, np.inf]])
        assert_raises(ValueError, histogramdd, vals, range=[[0.0, 1.0], [np.nan, 0.75], [0.25, 0.5]])

    def test_equal_edges(self):
        """ Test that adjacent entries in an edge array can be equal """
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        x_edges = np.array([0, 2, 2])
        y_edges = 1
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))
        hist_expected = np.array([[2.0], [1.0]])
        assert_equal(hist, hist_expected)

    def test_edge_dtype(self):
        """ Test that if an edge array is input, its type is preserved """
        x = np.array([0, 10, 20])
        y = x / 10
        x_edges = np.array([0, 5, 15, 20])
        y_edges = x_edges / 10
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))
        assert_equal(edges[0].dtype, x_edges.dtype)
        assert_equal(edges[1].dtype, y_edges.dtype)

    def test_large_integers(self):
        big = 2 ** 60
        x = np.array([0], np.int64)
        x_edges = np.array([-1, +1], np.int64)
        y = big + x
        y_edges = big + x_edges
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))
        assert_equal(hist[0, 0], 1)

    def test_density_non_uniform_2d(self):
        x_edges = np.array([0, 2, 8])
        y_edges = np.array([0, 6, 8])
        relative_areas = np.array([[3, 9], [1, 3]])
        x = np.array([1] + [1] * 3 + [7] * 3 + [7] * 9)
        y = np.array([7] + [1] * 3 + [7] * 3 + [1] * 9)
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges))
        assert_equal(hist, relative_areas)
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges), density=True)
        assert_equal(hist, 1 / (8 * 8))

    def test_density_non_uniform_1d(self):
        v = np.arange(10)
        bins = np.array([0, 1, 3, 6, 10])
        hist, edges = histogram(v, bins, density=True)
        hist_dd, edges_dd = histogramdd((v,), (bins,), density=True)
        assert_equal(hist, hist_dd)
        assert_equal(edges, edges_dd[0])