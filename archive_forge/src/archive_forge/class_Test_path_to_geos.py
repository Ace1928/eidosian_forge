from matplotlib.path import Path
import shapely.geometry as sgeom
import cartopy.mpl.patch as cpatch
class Test_path_to_geos:

    def test_empty_polygon(self):
        p = Path([[0, 0], [0, 0], [0, 0], [0, 0], [1, 2], [1, 2], [1, 2], [1, 2], [2, 3], [2, 3], [2, 3], [42, 42], [193.75, -14.166664123535156], [193.75, -14.166664123535158], [193.75, -14.166664123535156], [193.75, -14.166664123535156]], codes=[1, 2, 2, 79] * 4)
        geoms = cpatch.path_to_geos(p)
        assert [type(geom) for geom in geoms] == [sgeom.Point] * 4
        assert len(geoms) == 4

    def test_non_polygon_loop(self):
        p = Path([[0, 10], [170, 20], [-170, 30], [0, 10]], codes=[1, 2, 2, 2])
        geoms = cpatch.path_to_geos(p)
        assert [type(geom) for geom in geoms] == [sgeom.MultiLineString]
        assert len(geoms) == 1

    def test_polygon_with_interior_and_singularity(self):
        p = Path([[0, -90], [200, -40], [200, 40], [0, 40], [0, -90], [126, 26], [126, 26], [126, 26], [126, 26], [126, 26], [114, 5], [103, 8], [126, 12], [126, 0], [114, 5]], codes=[1, 2, 2, 2, 79, 1, 2, 2, 2, 79, 1, 2, 2, 2, 79])
        geoms = cpatch.path_to_geos(p)
        assert [type(geom) for geom in geoms] == [sgeom.Polygon, sgeom.Point]
        assert len(geoms[0].interiors) == 1

    def test_nested_polygons(self):
        vertices = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0], [2, 2], [2, 8], [8, 8], [8, 2], [2, 2], [4, 4], [4, 6], [6, 6], [6, 4], [4, 4]]
        codes = [1, 2, 2, 2, 79, 1, 2, 2, 2, 79, 1, 2, 2, 2, 79]
        p = Path(vertices, codes=codes)
        geoms = cpatch.path_to_geos(p)
        assert len(geoms) == 2
        assert all((isinstance(geom, sgeom.Polygon) for geom in geoms))
        assert len(geoms[0].interiors) == 1
        assert len(geoms[1].interiors) == 0