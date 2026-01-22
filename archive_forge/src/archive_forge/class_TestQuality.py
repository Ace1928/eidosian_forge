import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
class TestQuality:

    def setup_class(self):
        projection = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
        polygon = sgeom.Polygon([(177.5, -57.38460319), (180.1, -57.445077), (175.0, -57.19913331)])
        self.multi_polygon = projection.project_geometry(polygon)

    def test_split(self):
        assert len(self.multi_polygon.geoms) == 2

    def test_repeats(self):
        xy = np.array(self.multi_polygon.geoms[0].exterior.coords)
        same = (xy[1:] == xy[:-1]).all(axis=1)
        assert not any(same), 'Repeated points in projected geometry.'

    def test_symmetry(self):
        xy = np.array(self.multi_polygon.geoms[0].exterior.coords)
        boundary = np.logical_or(xy[:, 1] == 90, xy[:, 1] == -90)
        regions = (boundary[1:] != boundary[:-1]).cumsum()
        regions = np.insert(regions, 0, 0)
        for i in range(int(boundary[0]), regions.max(), 2):
            indices = np.where(regions == i)
            x = xy[indices, 0]
            delta = np.diff(x)
            num_incr = np.count_nonzero(delta > 0)
            num_decr = np.count_nonzero(delta < 0)
            assert abs(num_incr - num_decr) < 3, 'Too much asymmetry.'