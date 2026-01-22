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
class SouthPolarStereo(Stereographic):

    def __init__(self, central_longitude=0.0, true_scale_latitude=None, globe=None):
        super().__init__(central_latitude=-90, central_longitude=central_longitude, true_scale_latitude=true_scale_latitude, globe=globe)