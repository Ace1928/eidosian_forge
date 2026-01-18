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
def to_authority(self, auth_name: Optional[str]=None, min_confidence: int=70):
    """
        .. versionadded:: 2.2.0

        Return the authority name and code best matching the CRS
        or None if it a match is not found.

        Example:

        >>> from pyproj import CRS
        >>> ccs = CRS("EPSG:4328")
        >>> ccs.to_authority()
        ('EPSG', '4328')

        If the CRS is bound, you can get an authority from
        the source CRS:

        >>> from pyproj import CRS
        >>> ccs = CRS("+proj=geocent +datum=WGS84 +towgs84=0,0,0")
        >>> ccs.to_authority()
        >>> ccs.source_crs.to_authority()
        ('EPSG', '4978')
        >>> ccs == CRS.from_authorty('EPSG', '4978')
        False

        Parameters
        ----------
        auth_name: str, optional
            The name of the authority to filter by.
        min_confidence: int, default=70
            A value between 0-100 where 100 is the most confident.
            :ref:`min_confidence`

        Returns
        -------
        tuple(str, str) or None:
            The best matching (<auth_name>, <code>) for the confidence level.
        """
    return self._crs.to_authority(auth_name=auth_name, min_confidence=min_confidence)