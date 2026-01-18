import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_plate_carree_partial_wrap(self):
    proj = ccrs.PlateCarree()
    poly = sgeom.box(170, 0, 190, 10)
    multi_polygon = proj.project_geometry(poly, proj)
    assert len(multi_polygon.geoms) == 2
    poly1, poly2 = multi_polygon.geoms
    if 170.0 not in poly1.bounds:
        poly1, poly2 = (poly2, poly1)
    self._assert_bounds(poly1.bounds, 170, 0, 180, 10)
    self._assert_bounds(poly2.bounds, -180, 0, -170, 10)