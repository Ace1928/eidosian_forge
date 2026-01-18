import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_stitch(self):
    coords = [(0.0, -70.70499926182919), (0.0, -71.25), (0.0, -72.5), (0.0, -73.49076371657017), (360.0, -73.49076371657017), (360.0, -72.5), (360.0, -71.25), (360.0, -70.70499926182919), (350, -73), (10, -73)]
    src_proj = ccrs.PlateCarree()
    target_proj = ccrs.Stereographic(80)
    linear_ring = sgeom.LinearRing(coords)
    rings, mlinestr = target_proj.project_geometry(linear_ring, src_proj)
    assert len(mlinestr.geoms) == 1
    assert len(rings) == 0
    linear_ring = sgeom.LinearRing(coords[::-1])
    rings, mlinestr = target_proj.project_geometry(linear_ring, src_proj)
    assert len(mlinestr.geoms) == 1
    assert len(rings) == 0