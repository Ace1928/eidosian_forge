import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_plate_carree_no_wrap(self):
    proj = ccrs.PlateCarree()
    poly = sgeom.box(0, 0, 10, 10)
    multi_polygon = proj.project_geometry(poly, proj)
    assert len(multi_polygon.geoms) == 1
    polygon = multi_polygon.geoms[0]
    self._assert_bounds(polygon.bounds, 0, 0, 10, 10)