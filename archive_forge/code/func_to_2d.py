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
def to_2d(self, name: Optional[str]=None) -> 'CRS':
    """
        .. versionadded:: 3.6.0

        Convert the current CRS to the 2D version if it makes sense.

        Parameters
        ----------
        name: str, optional
            CRS name. Defaults to use the name of the original CRS.

        Returns
        -------
        CRS
        """
    return self.__class__(self._crs.to_2d(name=name))