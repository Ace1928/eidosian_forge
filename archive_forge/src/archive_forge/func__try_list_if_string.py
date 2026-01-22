import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _try_list_if_string(input_str):
    """
    Attempt to convert string to list if it is a string
    """
    if not isinstance(input_str, str):
        return input_str
    val_split = input_str.split(',')
    if len(val_split) > 1:
        return [float(sval.strip()) for sval in val_split]
    return input_str