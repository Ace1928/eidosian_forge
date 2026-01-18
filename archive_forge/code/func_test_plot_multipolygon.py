import pytest
from numpy.testing import assert_allclose
from shapely import box, get_coordinates, LineString, MultiLineString, Point
from shapely.plotting import patch_from_polygon, plot_line, plot_points, plot_polygon
def test_plot_multipolygon():
    poly = box(0, 0, 1, 1).union(box(2, 2, 3, 3))
    artist, _ = plot_polygon(poly)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(poly))