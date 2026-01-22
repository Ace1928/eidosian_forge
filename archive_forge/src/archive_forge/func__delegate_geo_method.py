from warnings import warn
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from shapely.geometry import box, MultiPoint
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
from .array import GeometryArray, GeometryDtype, points_from_xy
def _delegate_geo_method(op, this, *args, **kwargs):
    """Unary operation that returns a GeoSeries"""
    from .geoseries import GeoSeries
    a_this = GeometryArray(this.geometry.values)
    data = getattr(a_this, op)(*args, **kwargs)
    return GeoSeries(data, index=this.index, crs=this.crs)