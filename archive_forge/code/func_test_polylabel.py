import unittest
import pytest
from shapely.algorithms.polylabel import Cell, polylabel
from shapely.errors import TopologicalError
from shapely.geometry import LineString, Point, Polygon
def test_polylabel(self):
    """
        Finds pole of inaccessibility for a polygon with a tolerance of 10

        """
    polygon = LineString([(0, 0), (50, 200), (100, 100), (20, 50), (-100, -20), (-150, -200)]).buffer(100)
    label = polylabel(polygon, tolerance=10)
    expected = Point(59.35615556364569, 121.8391962974644)
    assert expected.equals_exact(label, 1e-06)