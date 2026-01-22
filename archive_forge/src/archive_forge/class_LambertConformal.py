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
class LambertConformal(Projection):
    """
    A Lambert Conformal conic projection.

    """

    def __init__(self, central_longitude=-96.0, central_latitude=39.0, false_easting=0.0, false_northing=0.0, standard_parallels=(33, 45), globe=None, cutoff=-30):
        """
        Parameters
        ----------
        central_longitude: optional
            The central longitude. Defaults to -96.
        central_latitude: optional
            The central latitude. Defaults to 39.
        false_easting: optional
            X offset from planar origin in metres. Defaults to 0.
        false_northing: optional
            Y offset from planar origin in metres. Defaults to 0.
        standard_parallels: optional
            Standard parallel latitude(s). Defaults to (33, 45).
        globe: optional
            A :class:`cartopy.crs.Globe`. If omitted, a default globe is
            created.
        cutoff: optional
            Latitude of map cutoff.
            The map extends to infinity opposite the central pole
            so we must cut off the map drawing before then.
            A value of 0 will draw half the globe. Defaults to -30.

        """
        proj4_params = [('proj', 'lcc'), ('lon_0', central_longitude), ('lat_0', central_latitude), ('x_0', false_easting), ('y_0', false_northing)]
        n_parallels = len(standard_parallels)
        if not 1 <= n_parallels <= 2:
            raise ValueError(f'1 or 2 standard parallels must be specified. Got {n_parallels} ({standard_parallels})')
        proj4_params.append(('lat_1', standard_parallels[0]))
        if n_parallels == 2:
            proj4_params.append(('lat_2', standard_parallels[1]))
        super().__init__(proj4_params, globe=globe)
        if n_parallels == 1:
            plat = 90 if standard_parallels[0] > 0 else -90
        else:
            if abs(standard_parallels[0]) > abs(standard_parallels[1]):
                poliest_sec = standard_parallels[0]
            else:
                poliest_sec = standard_parallels[1]
            plat = 90 if poliest_sec > 0 else -90
        self.cutoff = cutoff
        n = 91
        lons = np.empty(n + 2)
        lats = np.full(n + 2, float(cutoff))
        lons[0] = lons[-1] = 0
        lats[0] = lats[-1] = plat
        if plat == 90:
            lons[1:-1] = np.linspace(central_longitude + 180 - 0.001, central_longitude - 180 + 0.001, n)
        else:
            lons[1:-1] = np.linspace(central_longitude - 180 + 0.001, central_longitude + 180 - 0.001, n)
        points = self.transform_points(PlateCarree(globe=globe), lons, lats)
        self._boundary = sgeom.LinearRing(points)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        self._x_limits = (mins[0], maxs[0])
        self._y_limits = (mins[1], maxs[1])
        self.threshold = 100000.0

    def __eq__(self, other):
        res = super().__eq__(other)
        if hasattr(other, 'cutoff'):
            res = res and self.cutoff == other.cutoff
        return res

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.proj4_init, self.cutoff))

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits