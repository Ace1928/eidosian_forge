import threading
import warnings
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Optional, Union, overload
from pyproj import CRS
from pyproj._compat import cstrencode
from pyproj._crs import AreaOfUse, CoordinateOperation
from pyproj._datadir import _clear_proj_error
from pyproj._transformer import (  # noqa: F401 pylint: disable=unused-import
from pyproj.datadir import get_user_data_dir
from pyproj.enums import ProjVersion, TransformDirection, WktVersion
from pyproj.exceptions import ProjError
from pyproj.sync import _download_resource_file
from pyproj.utils import _convertback, _copytobuffer
def transform_bounds(self, left: float, bottom: float, right: float, top: float, densify_pts: int=21, radians: bool=False, errcheck: bool=False, direction: Union[TransformDirection, str]=TransformDirection.FORWARD) -> tuple[float, float, float, float]:
    """
        .. versionadded:: 3.1.0

        See: :c:func:`proj_trans_bounds`

        Transform boundary densifying the edges to account for nonlinear
        transformations along these edges and extracting the outermost bounds.

        If the destination CRS is geographic and right < left then the bounds
        crossed the antimeridian. In this scenario there are two polygons,
        one on each side of the antimeridian. The first polygon should be
        constructed with (left, bottom, 180, top) and the second with
        (-180, bottom, top, right).

        To construct the bounding polygons with shapely::

            def bounding_polygon(left, bottom, right, top):
                if right < left:
                    return shapely.geometry.MultiPolygon(
                        [
                            shapely.geometry.box(left, bottom, 180, top),
                            shapely.geometry.box(-180, bottom, right, top),
                        ]
                    )
                return shapely.geometry.box(left, bottom, right, top)


        Parameters
        ----------
        left: float
            Minimum bounding coordinate of the first axis in source CRS
            (or the target CRS if using the reverse direction).
        bottom: float
            Minimum bounding coordinate of the second axis in source CRS.
            (or the target CRS if using the reverse direction).
        right: float
            Maximum bounding coordinate of the first axis in source CRS.
            (or the target CRS if using the reverse direction).
        top: float
            Maximum bounding coordinate of the second axis in source CRS.
            (or the target CRS if using the reverse direction).
        densify_points: uint, default=21
            Number of points to add to each edge to account for nonlinear edges
            produced by the transform process. Large numbers will produce worse
            performance.
        radians: bool, default=False
            If True, will expect input data to be in radians and will return radians
            if the projection is geographic. Otherwise, it uses degrees.
        errcheck: bool, default=False
            If True, an exception is raised if the errors are found in the process.
            If False, ``inf`` is returned for errors.
        direction: pyproj.enums.TransformDirection, optional
            The direction of the transform.
            Default is :attr:`pyproj.enums.TransformDirection.FORWARD`.

        Returns
        -------
        left, bottom, right, top: float
            Outermost coordinates in target coordinate reference system.
        """
    return self._transformer._transform_bounds(left=left, bottom=bottom, right=right, top=top, densify_pts=densify_pts, radians=radians, errcheck=errcheck, direction=direction)