import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_self_intersecting_1(self):
    wkt = 'POLYGON ((366.22000122 -9.71489298, 366.73212393 -9.679999349999999, 366.77412634 -8.767753000000001, 366.17762962 -9.679999349999999, 366.22000122 -9.71489298), (366.22000122 -9.692636309999999, 366.32998657 -9.603356099999999, 366.74765799 -9.019999500000001, 366.5094086 -9.63175386, 366.22000122 -9.692636309999999))'
    geom = shapely.wkt.loads(wkt)
    source, target = (ccrs.RotatedPole(198.0, 39.25), ccrs.EuroPP())
    projected = target.project_geometry(geom, source)
    area = projected.area
    assert 2200000000.0 < area < 2300000000.0, f'Got area {area}, expecting ~2.2e9'