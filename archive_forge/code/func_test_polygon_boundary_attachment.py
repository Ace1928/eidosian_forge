import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_polygon_boundary_attachment(self):
    polygon = sgeom.Polygon([(-10, 30), (10, 60), (10, 50)])
    projection = ccrs.Robinson(170.6)
    projection.project_geometry(polygon)