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
class EckertII(_Eckert):
    """
    An Eckert II projection.

    This projection is pseudocylindrical, and equal-area. Both meridians and
    parallels are straight lines. Its non-equal-area pair with equally-spaced
    parallels is :class:`EckertI`.

    """
    _proj_name = 'eck2'