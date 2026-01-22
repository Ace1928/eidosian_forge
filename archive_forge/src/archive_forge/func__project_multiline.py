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
def _project_multiline(self, geometry, src_crs):
    geoms = []
    for geom in geometry.geoms:
        r = self._project_line_string(geom, src_crs)
        if r:
            geoms.extend(r.geoms)
    if geoms:
        return sgeom.MultiLineString(geoms)
    else:
        return []