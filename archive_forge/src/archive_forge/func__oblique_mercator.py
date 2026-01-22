import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _oblique_mercator(cf_params):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_oblique_mercator
    """
    return HotineObliqueMercatorBConversion(latitude_projection_centre=cf_params['latitude_of_projection_origin'], longitude_projection_centre=cf_params['longitude_of_projection_origin'], azimuth_initial_line=cf_params['azimuth_of_central_line'], angle_from_rectified_to_skew_grid=0.0, scale_factor_on_initial_line=cf_params.get('scale_factor_at_projection_origin', 1.0), easting_projection_centre=cf_params.get('false_easting', 0.0), northing_projection_centre=cf_params.get('false_northing', 0.0))