import numbers
import operator
import warnings
import inspect
from functools import lru_cache
import numpy as np
import pandas as pd
from pandas.api.extensions import (
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import BaseGeometry
import shapely.ops
import shapely.wkt
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from . import _compat as compat
from . import _vectorized as vectorized
from .sindex import _get_sindex_class
class GeometryDtype(ExtensionDtype):
    type = BaseGeometry
    name = 'geometry'
    na_value = np.nan

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError("'construct_from_string' expects a string, got {}".format(type(string)))
        elif string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from '{}'".format(cls.__name__, string))

    @classmethod
    def construct_array_type(cls):
        return GeometryArray