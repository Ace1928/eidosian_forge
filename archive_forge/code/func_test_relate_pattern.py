import unittest
import pytest
import shapely
from shapely.geometry import Point, Polygon
def test_relate_pattern(self):
    g1 = Polygon([(0, 0), (0, 1), (3, 1), (3, 0), (0, 0)])
    g2 = Polygon([(1, -1), (1, 2), (2, 2), (2, -1), (1, -1)])
    g3 = Point(5, 5)
    assert g1.relate(g2) == '212101212'
    assert g1.relate_pattern(g2, '212101212')
    assert g1.relate_pattern(g2, '*********')
    assert g1.relate_pattern(g2, '2********')
    assert g1.relate_pattern(g2, 'T********')
    assert not g1.relate_pattern(g2, '112101212')
    assert not g1.relate_pattern(g2, '1********')
    assert g1.relate_pattern(g3, 'FF2FF10F2')
    with pytest.raises(shapely.GEOSException, match='IllegalArgumentException'):
        g1.relate_pattern(g2, 'fail')