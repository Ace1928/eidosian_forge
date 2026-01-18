from matplotlib.path import Path
import shapely.geometry as sgeom
import cartopy.mpl.patch as cpatch
def test_polygon_with_interior_and_singularity(self):
    p = Path([[0, -90], [200, -40], [200, 40], [0, 40], [0, -90], [126, 26], [126, 26], [126, 26], [126, 26], [126, 26], [114, 5], [103, 8], [126, 12], [126, 0], [114, 5]], codes=[1, 2, 2, 2, 79, 1, 2, 2, 2, 79, 1, 2, 2, 2, 79])
    geoms = cpatch.path_to_geos(p)
    assert [type(geom) for geom in geoms] == [sgeom.Polygon, sgeom.Point]
    assert len(geoms[0].interiors) == 1