import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
@pytest.mark.xfail(reason='TODO issue to CCW')
def test_polygon_shell_ccw_fully_on_rectangle_boundary():
    """Polygon shell (CCW) fully on rectangle boundary"""
    geom1 = load_wkt('POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))')
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == 'POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))'