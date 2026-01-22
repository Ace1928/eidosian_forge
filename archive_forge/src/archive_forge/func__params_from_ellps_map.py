import math
import warnings
from typing import Any, Optional, Union
from pyproj._geod import Geod as _Geod
from pyproj._geod import GeodIntermediateReturn, geodesic_version_str
from pyproj._geod import reverse_azimuth as _reverse_azimuth
from pyproj.enums import GeodIntermediateFlag
from pyproj.exceptions import GeodError
from pyproj.list import get_ellps_map
from pyproj.utils import DataType, _convertback, _copytobuffer
def _params_from_ellps_map(ellps: str) -> tuple[float, float, float, float, bool]:
    """
    Build Geodesic parameters from PROJ ellips map

    Parameter
    ---------
    ellps: str
        The name of the ellipse in the map.

    Returns
    -------
    tuple[float, float, float, float, bool]

    """
    ellps_dict = pj_ellps[ellps]
    semi_major_axis: float = ellps_dict['a']
    sphere = False
    if ellps_dict['description'] == 'Normal Sphere':
        sphere = True
    if 'b' in ellps_dict:
        semi_minor_axis: float = ellps_dict['b']
        eccentricity_squared: float = 1.0 - semi_minor_axis ** 2 / semi_major_axis ** 2
        flattening: float = (semi_major_axis - semi_minor_axis) / semi_major_axis
    elif 'rf' in ellps_dict:
        flattening = 1.0 / ellps_dict['rf']
        semi_minor_axis = semi_major_axis * (1.0 - flattening)
        eccentricity_squared = 1.0 - semi_minor_axis ** 2 / semi_major_axis ** 2
    return (semi_major_axis, semi_minor_axis, flattening, eccentricity_squared, sphere)