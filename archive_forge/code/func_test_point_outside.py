import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
def test_point_outside():
    """Point outside"""
    geom1 = load_wkt('POINT (0 0)')
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == 'GEOMETRYCOLLECTION EMPTY'