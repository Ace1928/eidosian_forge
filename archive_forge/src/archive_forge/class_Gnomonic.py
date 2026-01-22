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
class Gnomonic(Projection):
    _handles_ellipses = False

    def __init__(self, central_latitude=0.0, central_longitude=0.0, globe=None):
        proj4_params = [('proj', 'gnom'), ('lat_0', central_latitude), ('lon_0', central_longitude)]
        super().__init__(proj4_params, globe=globe)
        self._max = 50000000.0
        self.threshold = 100000.0

    @property
    def boundary(self):
        return sgeom.Point(0, 0).buffer(self._max).exterior

    @property
    def x_limits(self):
        return (-self._max, self._max)

    @property
    def y_limits(self):
        return (-self._max, self._max)