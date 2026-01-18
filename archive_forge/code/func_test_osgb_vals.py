import numpy as np
import pytest
import cartopy.crs as ccrs
def test_osgb_vals(self, approx):
    proj = ccrs.TransverseMercator(central_longitude=-2, central_latitude=49, scale_factor=0.9996012717, false_easting=400000, false_northing=-100000, globe=ccrs.Globe(datum='OSGB36', ellipse='airy'), approx=approx)
    res = proj.transform_point(*self.point_a, src_crs=self.src_crs)
    np.testing.assert_array_almost_equal(res, (295971.28668, 93064.27666), decimal=5)
    res = proj.transform_point(*self.point_b, src_crs=self.src_crs)
    np.testing.assert_array_almost_equal(res, (577274.9838, 69740.49227), decimal=5)