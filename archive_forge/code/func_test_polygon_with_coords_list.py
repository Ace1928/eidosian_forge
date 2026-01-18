import pytest
from shapely.geometry import MultiLineString, Polygon, shape
from shapely.geometry.geo import _is_coordinates_empty
def test_polygon_with_coords_list():
    geom = {'type': 'Polygon', 'coordinates': [[[5, 10], [10, 10], [10, 5]]]}
    obj = shape(geom)
    assert obj == Polygon([(5, 10), (10, 10), (10, 5)])