import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_same_points_on_boundary_1(self):
    source = ccrs.PlateCarree()
    target = ccrs.PlateCarree(central_longitude=180)
    geom = sgeom.Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)], [[(-10, 0), (0, 20), (10, 0), (0, -20)]])
    projected = target.project_geometry(geom, source)
    assert abs(1200 - projected.area) < 1e-05