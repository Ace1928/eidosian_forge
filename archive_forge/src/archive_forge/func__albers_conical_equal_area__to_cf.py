import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _albers_conical_equal_area__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_albers_equal_area

    """
    params = _to_dict(conversion)
    return {'grid_mapping_name': 'albers_conical_equal_area', 'standard_parallel': (params['latitude_of_1st_standard_parallel'], params['latitude_of_2nd_standard_parallel']), 'latitude_of_projection_origin': params['latitude_of_false_origin'], 'longitude_of_central_meridian': params['longitude_of_false_origin'], 'false_easting': params['easting_at_false_origin'], 'false_northing': params['northing_at_false_origin']}