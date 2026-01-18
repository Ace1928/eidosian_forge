import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_self_intersecting_2(self):
    wkt = 'POLYGON ((343 20, 345 23, 342 25, 343 22, 340 25, 341 25, 340 25, 343 20), (343 21, 343 22, 344 23, 343 21))'
    geom = shapely.wkt.loads(wkt)
    source = target = ccrs.RotatedPole(193.0, 41.0)
    projected = target.project_geometry(geom, source)
    assert 7.9 < projected.area < 8.1