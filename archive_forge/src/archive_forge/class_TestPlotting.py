import pytest
from numpy.testing import assert_, assert_array_equal, suppress_warnings
from scipy.spatial import \
@pytest.mark.skipif(not has_matplotlib, reason='Matplotlib not available')
class TestPlotting:
    points = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_delaunay(self):
        fig = plt.figure()
        obj = Delaunay(self.points)
        s_before = obj.simplices.copy()
        with suppress_warnings() as sup:
            sup.filter(message='The ishold function was deprecated in version')
            r = delaunay_plot_2d(obj, ax=fig.gca())
        assert_array_equal(obj.simplices, s_before)
        assert_(r is fig)
        delaunay_plot_2d(obj, ax=fig.gca())

    def test_voronoi(self):
        fig = plt.figure()
        obj = Voronoi(self.points)
        with suppress_warnings() as sup:
            sup.filter(message='The ishold function was deprecated in version')
            r = voronoi_plot_2d(obj, ax=fig.gca())
        assert_(r is fig)
        voronoi_plot_2d(obj)
        voronoi_plot_2d(obj, show_vertices=False)

    def test_convex_hull(self):
        fig = plt.figure()
        tri = ConvexHull(self.points)
        with suppress_warnings() as sup:
            sup.filter(message='The ishold function was deprecated in version')
            r = convex_hull_plot_2d(tri, ax=fig.gca())
        assert_(r is fig)
        convex_hull_plot_2d(tri)