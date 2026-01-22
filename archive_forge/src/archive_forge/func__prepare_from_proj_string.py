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
def _prepare_from_proj_string(in_crs_string: str) -> str:
    in_crs_string = re.sub('[\\s+]?=[\\s+]?', '=', in_crs_string.lstrip())
    starting_params = ('+init', '+proj', 'init', 'proj')
    if not in_crs_string.startswith(starting_params):
        kvpairs: list[str] = []
        first_item_inserted = False
        for kvpair in in_crs_string.split():
            if not first_item_inserted and kvpair.startswith(starting_params):
                kvpairs.insert(0, kvpair)
                first_item_inserted = True
            else:
                kvpairs.append(kvpair)
        in_crs_string = ' '.join(kvpairs)
    if 'type=crs' not in in_crs_string:
        if '+' in in_crs_string:
            in_crs_string += ' +type=crs'
        else:
            in_crs_string += ' type=crs'
    in_crs_string = in_crs_string.replace('+init=EPSG', '+init=epsg').strip()
    if in_crs_string.startswith(('+init', 'init')):
        warnings.warn("'+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6", FutureWarning, stacklevel=2)
    return in_crs_string