import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _polar_stereographic__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#polar-stereographic
    """
    params = _to_dict(conversion)
    if conversion.method_name.lower().endswith('(variant b)'):
        return {'grid_mapping_name': 'polar_stereographic', 'standard_parallel': params['latitude_of_standard_parallel'], 'straight_vertical_longitude_from_pole': params['longitude_of_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing']}
    return {'grid_mapping_name': 'polar_stereographic', 'latitude_of_projection_origin': params['latitude_of_natural_origin'], 'straight_vertical_longitude_from_pole': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing'], 'scale_factor_at_projection_origin': params['scale_factor_at_natural_origin']}