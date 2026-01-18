import pytest
from numpy.testing import assert_allclose
from shapely import box, get_coordinates, LineString, MultiLineString, Point
from shapely.plotting import patch_from_polygon, plot_line, plot_points, plot_polygon
def test_plot_polygon_with_interior():
    poly = box(0, 0, 1, 1).difference(box(0.2, 0.2, 0.5, 0.5))
    artist, _ = plot_polygon(poly)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(poly))