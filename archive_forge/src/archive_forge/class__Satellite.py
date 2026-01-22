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
class _Satellite(Projection):

    def __init__(self, projection, satellite_height=35785831, central_longitude=0.0, central_latitude=0.0, false_easting=0, false_northing=0, globe=None, sweep_axis=None):
        proj4_params = [('proj', projection), ('lon_0', central_longitude), ('lat_0', central_latitude), ('h', satellite_height), ('x_0', false_easting), ('y_0', false_northing), ('units', 'm')]
        if sweep_axis:
            proj4_params.append(('sweep', sweep_axis))
        super().__init__(proj4_params, globe=globe)

    def _set_boundary(self, coords):
        self._boundary = sgeom.LinearRing(coords.T)
        mins = np.min(coords, axis=1)
        maxs = np.max(coords, axis=1)
        self._x_limits = (mins[0], maxs[0])
        self._y_limits = (mins[1], maxs[1])
        self.threshold = np.diff(self._x_limits)[0] * 0.02

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits