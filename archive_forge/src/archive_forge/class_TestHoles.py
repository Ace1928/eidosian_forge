import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
class TestHoles(PolygonTests):

    def test_simple(self):
        proj = ccrs.PlateCarree()
        poly = sgeom.Polygon(ring(-40, -40, 40, 40, True), [ring(-20, -20, 20, 20, False)])
        multi_polygon = proj.project_geometry(poly)
        assert len(multi_polygon.geoms) == 1
        assert len(multi_polygon.geoms[0].interiors) == 1
        polygon = multi_polygon.geoms[0]
        self._assert_bounds(polygon.bounds, -40, -47, 40, 47)
        self._assert_bounds(polygon.interiors[0].bounds, -20, -21, 20, 21)

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

    def test_wrapped_poly_wrapped_hole(self):
        proj = ccrs.PlateCarree(-180)
        poly = sgeom.Polygon(ring(-40, -40, 40, 40, True), [ring(-20, -20, 20, 20, False)])
        multi_polygon = proj.project_geometry(poly)
        assert len(multi_polygon.geoms) == 2
        assert len(multi_polygon.geoms[0].interiors) == 0
        assert len(multi_polygon.geoms[1].interiors) == 0
        polygon = multi_polygon.geoms[0]
        self._assert_bounds(polygon.bounds, 140, -47, 180, 47)
        polygon = multi_polygon.geoms[1]
        self._assert_bounds(polygon.bounds, -180, -47, -140, 47)

    def test_inverted_poly_simple_hole(self):
        proj = ccrs.NorthPolarStereo()
        poly = sgeom.Polygon([(0, 0), (-90, 0), (-180, 0), (-270, 0)], [[(0, -30), (90, -30), (180, -30), (270, -30)]])
        multi_polygon = proj.project_geometry(poly)
        assert len(multi_polygon.geoms) == 1
        assert len(multi_polygon.geoms[0].interiors) == 1
        polygon = multi_polygon.geoms[0]
        self._assert_bounds(polygon.bounds, -24000000.0, -24000000.0, 24000000.0, 24000000.0, 1000000.0)
        self._assert_bounds(polygon.interiors[0].bounds, -12000000.0, -12000000.0, 12000000.0, 12000000.0, 1000000.0)

    def test_inverted_poly_clipped_hole(self):
        proj = ccrs.NorthPolarStereo()
        poly = sgeom.Polygon([(0, 0), (-90, 0), (-180, 0), (-270, 0)], [[(-135, -60), (-45, -60), (45, -60), (135, -60)]])
        multi_polygon = proj.project_geometry(poly)
        assert len(multi_polygon.geoms) == 1
        assert len(multi_polygon.geoms[0].interiors) == 1
        polygon = multi_polygon.geoms[0]
        self._assert_bounds(polygon.bounds, -50000000.0, -50000000.0, 50000000.0, 50000000.0, 1000000.0)
        self._assert_bounds(polygon.interiors[0].bounds, -12000000.0, -12000000.0, 12000000.0, 12000000.0, 1000000.0)
        assert abs(polygon.area - 7300000000000000.0) < 10000000000000.0

    def test_inverted_poly_removed_hole(self):
        proj = ccrs.NorthPolarStereo(globe=ccrs.Globe(ellipse='WGS84'))
        poly = sgeom.Polygon([(0, 0), (-90, 0), (-180, 0), (-270, 0)], [[(-135, -75), (-45, -75), (45, -75), (135, -75)]])
        multi_polygon = proj.project_geometry(poly)
        assert len(multi_polygon.geoms) == 1
        assert len(multi_polygon.geoms[0].interiors) == 1
        polygon = multi_polygon.geoms[0]
        self._assert_bounds(polygon.bounds, -50000000.0, -50000000.0, 50000000.0, 50000000.0, 1000000.0)
        self._assert_bounds(polygon.interiors[0].bounds, -12000000.0, -12000000.0, 12000000.0, 12000000.0, 1000000.0)
        assert abs(polygon.area - 7340000000000000.0) < 10000000000000.0

    def test_multiple_interiors(self):
        exterior = ring(0, 0, 12, 12, True)
        interiors = [ring(1, 1, 2, 2, False), ring(1, 8, 2, 9, False)]
        poly = sgeom.Polygon(exterior, interiors)
        target = ccrs.PlateCarree()
        source = ccrs.Geodetic()
        assert len(list(target.project_geometry(poly, source).geoms)) == 1