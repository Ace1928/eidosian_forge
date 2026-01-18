import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_wrapped_poly_simple_hole(self):
    proj = ccrs.PlateCarree(-150)
    poly = sgeom.Polygon(ring(-40, -40, 40, 40, True), [ring(-20, -20, 20, 20, False)])
    multi_polygon = proj.project_geometry(poly)
    assert len(multi_polygon.geoms) == 2
    poly1, poly2 = multi_polygon.geoms
    if not len(poly1.interiors) == 1:
        poly1, poly2 = (poly2, poly1)
    assert len(poly1.interiors) == 1
    assert len(poly2.interiors) == 0
    self._assert_bounds(poly1.bounds, 110, -47, 180, 47)
    self._assert_bounds(poly1.interiors[0].bounds, 130, -21, 170, 21)
    self._assert_bounds(poly2.bounds, -180, -43, -170, 43)