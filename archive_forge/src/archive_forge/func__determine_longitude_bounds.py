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
def _determine_longitude_bounds(self, central_longitude):
    epsilon = 1e-10
    minlon = -180 + central_longitude
    maxlon = 180 + central_longitude
    if central_longitude > 0:
        maxlon -= epsilon
    elif central_longitude < 0:
        minlon += epsilon
    return (minlon, maxlon)