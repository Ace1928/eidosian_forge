import unittest
import pytest
from shapely.algorithms.polylabel import Cell, polylabel
from shapely.errors import TopologicalError
from shapely.geometry import LineString, Point, Polygon
def test_rectangle_special_case(self):
    """
        The centroid algorithm used is vulnerable to floating point errors
        and can give unexpected results for rectangular polygons. Test
        that this special case is handled correctly.
        https://github.com/mapbox/polylabel/issues/3
        """
    polygon = Polygon([(32.71997, -117.1931), (32.71997, -117.21065), (32.72408, -117.21065), (32.72408, -117.1931)])
    label = polylabel(polygon)
    assert label.coords[:] == [(32.722025, -117.201875)]