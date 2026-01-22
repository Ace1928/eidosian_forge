import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _lambert_cylindrical_equal_area(cf_params):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_lambert_cylindrical_equal_area
    """
    if 'scale_factor_at_projection_origin' in cf_params:
        return LambertCylindricalEqualAreaScaleConversion(scale_factor_natural_origin=cf_params['scale_factor_at_projection_origin'], longitude_natural_origin=cf_params.get('longitude_of_central_meridian', 0.0), false_easting=cf_params.get('false_easting', 0.0), false_northing=cf_params.get('false_northing', 0.0))
    return LambertCylindricalEqualAreaConversion(latitude_first_parallel=cf_params.get('standard_parallel', 0.0), longitude_natural_origin=cf_params.get('longitude_of_central_meridian', 0.0), false_easting=cf_params.get('false_easting', 0.0), false_northing=cf_params.get('false_northing', 0.0))