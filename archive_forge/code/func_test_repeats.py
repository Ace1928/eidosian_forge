import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_repeats(self):
    xy = np.array(self.multi_polygon.geoms[0].exterior.coords)
    same = (xy[1:] == xy[:-1]).all(axis=1)
    assert not any(same), 'Repeated points in projected geometry.'