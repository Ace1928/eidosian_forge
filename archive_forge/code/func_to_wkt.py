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
def to_wkt(self, version: Union[WktVersion, str]=WktVersion.WKT2_2019, pretty: bool=False, output_axis_rule: Optional[bool]=None) -> str:
    """
        Convert the projection to a WKT string.

        Version options:
          - WKT2_2015
          - WKT2_2015_SIMPLIFIED
          - WKT2_2019
          - WKT2_2019_SIMPLIFIED
          - WKT1_GDAL
          - WKT1_ESRI

        .. versionadded:: 3.6.0 output_axis_rule

        Parameters
        ----------
        version: pyproj.enums.WktVersion, optional
            The version of the WKT output.
            Default is :attr:`pyproj.enums.WktVersion.WKT2_2019`.
        pretty: bool, default=False
            If True, it will set the output to be a multiline string.
        output_axis_rule: bool, optional, default=None
            If True, it will set the axis rule on any case. If false, never.
            None for AUTO, that depends on the CRS and version.

        Returns
        -------
        str
        """
    wkt = self._crs.to_wkt(version=version, pretty=pretty, output_axis_rule=output_axis_rule)
    if wkt is None:
        raise CRSError(f"CRS cannot be converted to a WKT string of a '{version}' version. Select a different version of a WKT string or edit your CRS.")
    return wkt