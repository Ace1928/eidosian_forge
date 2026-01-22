import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _rotated_latitude_longitude(cf_params):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_rotated_pole
    """
    return PoleRotationNetCDFCFConversion(grid_north_pole_latitude=cf_params['grid_north_pole_latitude'], grid_north_pole_longitude=cf_params['grid_north_pole_longitude'], north_pole_grid_longitude=cf_params.get('north_pole_grid_longitude', 0.0))