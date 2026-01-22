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
class PlateCarree(_CylindricalProjection):

    def __init__(self, central_longitude=0.0, globe=None):
        globe = globe or Globe(semimajor_axis=WGS84_SEMIMAJOR_AXIS)
        proj4_params = [('proj', 'eqc'), ('lon_0', central_longitude), ('to_meter', math.radians(1) * (globe.semimajor_axis or WGS84_SEMIMAJOR_AXIS)), ('vto_meter', 1)]
        x_max = 180
        y_max = 90
        self.threshold = x_max / 360
        super().__init__(proj4_params, x_max, y_max, globe=globe)

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

    def quick_vertices_transform(self, vertices, src_crs):
        return_value = super().quick_vertices_transform(vertices, src_crs)
        if return_value is None and isinstance(src_crs, PlateCarree):
            self_params = self.proj4_params.copy()
            src_params = src_crs.proj4_params.copy()
            (self_params.pop('lon_0'), src_params.pop('lon_0'))
            xs, ys = (vertices[:, 0], vertices[:, 1])
            potential = self_params == src_params and self.y_limits[0] <= ys.min() and (self.y_limits[1] >= ys.max())
            if potential:
                mod = np.diff(src_crs.x_limits)[0]
                bboxes, proj_offset = self._bbox_and_offset(src_crs)
                x_lim = (xs.min(), xs.max())
                for poly in bboxes:
                    for i in [-1, 0, 1, 2]:
                        offset = mod * i - proj_offset
                        if poly[0] + offset <= x_lim[0] and poly[1] + offset >= x_lim[1]:
                            return_value = vertices + [[-offset, 0]]
                            break
                    if return_value is not None:
                        break
        return return_value