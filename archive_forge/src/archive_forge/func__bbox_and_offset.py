from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def _bbox_and_offset(self, other_plate_carree):
    """
        Return a pair of (xmin, xmax) pairs and an offset which can be used
        for identification of whether data in ``other_plate_carree`` needs
        to be transformed to wrap appropriately.

        >>> import cartopy.crs as ccrs
        >>> src = ccrs.PlateCarree(central_longitude=10)
        >>> bboxes, offset = ccrs.PlateCarree()._bbox_and_offset(src)
        >>> print(bboxes)
        [[-180, -170.0], [-170.0, 180]]
        >>> print(offset)
        10.0

        The returned values are longitudes in ``other_plate_carree``'s
        coordinate system.

        Warning
        -------
            The two CRSs must be identical in every way, other than their
            central longitudes. No checking of this is done.

        """
    self_lon_0 = self.proj4_params['lon_0']
    other_lon_0 = other_plate_carree.proj4_params['lon_0']
    lon_0_offset = other_lon_0 - self_lon_0
    lon_lower_bound_0 = self.x_limits[0]
    lon_lower_bound_1 = other_plate_carree.x_limits[0] + lon_0_offset
    if lon_lower_bound_1 < self.x_limits[0]:
        lon_lower_bound_1 += np.diff(self.x_limits)[0]
    lon_lower_bound_0, lon_lower_bound_1 = sorted([lon_lower_bound_0, lon_lower_bound_1])
    bbox = [[lon_lower_bound_0, lon_lower_bound_1], [lon_lower_bound_1, lon_lower_bound_0]]
    bbox[1][1] += np.diff(self.x_limits)[0]
    return (bbox, lon_0_offset)