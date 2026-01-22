import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
class TestBoundary:

    def test_no_polygon_boundary_reversal(self):
        polygon = sgeom.Polygon([(-10, 30), (10, 60), (10, 50)])
        projection = ccrs.Robinson(170.5)
        multi_polygon = projection.project_geometry(polygon)
        for polygon in multi_polygon.geoms:
            assert polygon.is_valid

    def test_polygon_boundary_attachment(self):
        polygon = sgeom.Polygon([(-10, 30), (10, 60), (10, 50)])
        projection = ccrs.Robinson(170.6)
        projection.project_geometry(polygon)

    def test_out_of_bounds(self):
        projection = ccrs.TransverseMercator(central_longitude=0, approx=True)
        polys = [([(86, -1), (86, 1), (88, 1), (88, -1)], 1), ([(86, -1), (86, 1), (130, 1), (88, -1)], 1), ([(86, -1), (86, 1), (130, 1), (130, -1)], 1), ([(120, -1), (120, 1), (130, 1), (130, -1)], 0)]
        for coords, expected_polys in polys:
            polygon = sgeom.Polygon(coords)
            multi_polygon = projection.project_geometry(polygon)
            assert len(multi_polygon.geoms) == expected_polys