import json
import re
import threading
import warnings
from typing import Any, Callable, Optional, Union
from pyproj._crs import (
from pyproj.crs._cf1x8 import (
from pyproj.crs.coordinate_operation import ToWGS84Transformation
from pyproj.crs.coordinate_system import Cartesian2DCS, Ellipsoidal2DCS, VerticalCS
from pyproj.enums import ProjVersion, WktVersion
from pyproj.exceptions import CRSError
from pyproj.geod import Geod
def _is_epsg_code(auth_code: Any) -> bool:
    if isinstance(auth_code, int):
        return True
    if isinstance(auth_code, str) and auth_code.isnumeric():
        return True
    if hasattr(auth_code, 'shape') and auth_code.shape == ():
        return True
    return False