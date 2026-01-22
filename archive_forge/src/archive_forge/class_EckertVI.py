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
class EckertVI(_Eckert):
    """
    An Eckert VI projection.

    This projection is pseudocylindrical, and equal-area. Parallels are
    unequally-spaced straight lines, while meridians are sinusoidal arcs. Its
    non-equal-area pair with equally-spaced parallels is :class:`EckertV`.

    It is commonly used for world maps.

    """
    _proj_name = 'eck6'