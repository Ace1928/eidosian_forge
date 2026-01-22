import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _pole_rotation_netcdf__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_rotated_pole

    https://github.com/OSGeo/PROJ/pull/2835
    """
    params = _to_dict(conversion)
    return {'grid_mapping_name': 'rotated_latitude_longitude', 'grid_north_pole_latitude': params['grid_north_pole_latitude_(netcdf_cf_convention)'], 'grid_north_pole_longitude': params['grid_north_pole_longitude_(netcdf_cf_convention)'], 'north_pole_grid_longitude': params['north_pole_grid_longitude_(netcdf_cf_convention)']}