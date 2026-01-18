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
def to_epsg(self, min_confidence: int=70) -> Optional[int]:
    """
        Return the EPSG code best matching the CRS
        or None if it a match is not found.

        Example:

        >>> from pyproj import CRS
        >>> ccs = CRS("EPSG:4328")
        >>> ccs.to_epsg()
        4328

        If the CRS is bound, you can attempt to get an epsg code from
        the source CRS:

        >>> from pyproj import CRS
        >>> ccs = CRS("+proj=geocent +datum=WGS84 +towgs84=0,0,0")
        >>> ccs.to_epsg()
        >>> ccs.source_crs.to_epsg()
        4978
        >>> ccs == CRS.from_epsg(4978)
        False

        Parameters
        ----------
        min_confidence: int, default=70
            A value between 0-100 where 100 is the most confident.
            :ref:`min_confidence`


        Returns
        -------
        Optional[int]:
            The best matching EPSG code matching the confidence level.
        """
    return self._crs.to_epsg(min_confidence=min_confidence)