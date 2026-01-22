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
def _crs_mismatch_warn(left, right, stacklevel=3):
    """
    Raise a CRS mismatch warning with the information on the assigned CRS.
    """
    if left.crs:
        left_srs = left.crs.to_string()
        left_srs = left_srs if len(left_srs) <= 50 else ' '.join([left_srs[:50], '...'])
    else:
        left_srs = None
    if right.crs:
        right_srs = right.crs.to_string()
        right_srs = right_srs if len(right_srs) <= 50 else ' '.join([right_srs[:50], '...'])
    else:
        right_srs = None
    warnings.warn('CRS mismatch between the CRS of left geometries and the CRS of right geometries.\nUse `to_crs()` to reproject one of the input geometries to match the CRS of the other.\n\nLeft CRS: {0}\nRight CRS: {1}\n'.format(left_srs, right_srs), UserWarning, stacklevel=stacklevel)