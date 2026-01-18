from numpy.testing import assert_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_sweep(self):
    geos = ccrs.Geostationary(sweep_axis='x')
    other_args = {'ellps=WGS84', 'h=35785831', 'lat_0=0.0', 'lon_0=0.0', 'sweep=x', 'units=m', 'x_0=0', 'y_0=0'}
    check_proj_params(self.expected_proj_name, geos, other_args)
    pt = geos.transform_point(-60, 25, ccrs.PlateCarree())
    assert_almost_equal(pt, (-4529521.6442, 2437479.4195), decimal=4)