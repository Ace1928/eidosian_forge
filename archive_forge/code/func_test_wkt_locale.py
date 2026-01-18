import locale
import sys
import unittest
from shapely.wkt import dumps, loads
def test_wkt_locale(self):
    p = loads('POINT (0.0 0.0)')
    assert p.x == 0.0
    assert p.y == 0.0
    wkt = dumps(p)
    assert wkt.startswith('POINT')
    assert ',' not in wkt