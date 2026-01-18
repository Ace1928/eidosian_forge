import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_tiny_point_between_boundary_points(self):
    target = ccrs.Orthographic(0, -75)
    source = ccrs.PlateCarree()
    wkt = 'POLYGON ((132 -40, 133 -6, 125.3 1, 115 -6, 132 -40))'
    geom = shapely.wkt.loads(wkt)
    target = ccrs.Orthographic(central_latitude=90.0, central_longitude=0)
    source = ccrs.PlateCarree()
    projected = target.project_geometry(geom, source)
    area = projected.area
    assert 81330 < area < 81340, f'Got area {area}, expecting ~81336'