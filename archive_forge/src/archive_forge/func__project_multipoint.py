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
def _project_multipoint(self, geometry, src_crs):
    geoms = []
    for geom in geometry.geoms:
        geoms.append(self._project_point(geom, src_crs))
    if geoms:
        return sgeom.MultiPoint(geoms)
    else:
        return sgeom.MultiPoint()