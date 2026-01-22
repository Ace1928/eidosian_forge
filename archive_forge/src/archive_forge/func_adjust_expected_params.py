from numpy.testing import assert_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def adjust_expected_params(self, expected):
    if self.expected_proj_name == 'geos':
        expected.add('sweep=y')