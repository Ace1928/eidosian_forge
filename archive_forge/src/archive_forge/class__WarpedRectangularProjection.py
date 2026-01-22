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
class _WarpedRectangularProjection(Projection, metaclass=ABCMeta):
    _wrappable = True

    def __init__(self, proj4_params, central_longitude, false_easting=None, false_northing=None, globe=None):
        if false_easting is not None:
            proj4_params += [('x_0', false_easting)]
        if false_northing is not None:
            proj4_params += [('y_0', false_northing)]
        super().__init__(proj4_params, globe=globe)
        minlon, maxlon = self._determine_longitude_bounds(central_longitude)
        n = 91
        lon = np.empty(2 * n + 1)
        lat = np.empty(2 * n + 1)
        lon[:n] = minlon
        lat[:n] = np.linspace(-90, 90, n)
        lon[n:2 * n] = maxlon
        lat[n:2 * n] = np.linspace(90, -90, n)
        lon[-1] = minlon
        lat[-1] = -90
        points = self.transform_points(self.as_geodetic(), lon, lat)
        self._boundary = sgeom.LinearRing(points)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        self._x_limits = (mins[0], maxs[0])
        self._y_limits = (mins[1], maxs[1])

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits