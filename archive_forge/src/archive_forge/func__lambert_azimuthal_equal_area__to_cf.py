import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _lambert_azimuthal_equal_area__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#lambert-azimuthal-equal-area
    """
    params = _to_dict(conversion)
    return {'grid_mapping_name': 'lambert_azimuthal_equal_area', 'latitude_of_projection_origin': params['latitude_of_natural_origin'], 'longitude_of_projection_origin': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing']}