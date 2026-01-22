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
class Miller(_RectangularProjection):
    _handles_ellipses = False

    def __init__(self, central_longitude=0.0, globe=None):
        if globe is None:
            globe = Globe(semimajor_axis=WGS84_SEMIMAJOR_AXIS, ellipse=None)
        a = globe.semimajor_axis or WGS84_SEMIMAJOR_AXIS
        proj4_params = [('proj', 'mill'), ('lon_0', central_longitude)]
        super().__init__(proj4_params, a * np.pi, a * 2.303412543376391, globe=globe)