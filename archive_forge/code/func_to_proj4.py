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
def to_proj4(self, version: Union[ProjVersion, int]=ProjVersion.PROJ_5) -> str:
    """
        Convert the projection to a PROJ string.

        .. warning:: You will likely lose important projection
          information when converting to a PROJ string from
          another format. See:
          https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems  # noqa: E501

        Parameters
        ----------
        version: pyproj.enums.ProjVersion
            The version of the PROJ string output.
            Default is :attr:`pyproj.enums.ProjVersion.PROJ_4`.

        Returns
        -------
        str
        """
    proj = self._crs.to_proj4(version=version)
    if proj is None:
        raise CRSError('CRS cannot be converted to a PROJ string.')
    return proj