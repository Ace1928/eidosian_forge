import json
import warnings
import numpy as np
import pandas as pd
import shapely.errors
from pandas import DataFrame, Series
from pandas.core.accessor import CachedAccessor
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
from geopandas.array import GeometryArray, GeometryDtype, from_shapely, to_wkb, to_wkt
from geopandas.base import GeoPandasBase, is_geometry_type
from geopandas.geoseries import GeoSeries
import geopandas.io
from geopandas.explore import _explore
from . import _compat as compat
from ._decorator import doc
def _ensure_geometry(data, crs=None):
    """
    Ensure the data is of geometry dtype or converted to it.

    If input is a (Geo)Series, output is a GeoSeries, otherwise output
    is GeometryArray.

    If the input is a GeometryDtype with a set CRS, `crs` is ignored.
    """
    if is_geometry_type(data):
        if isinstance(data, Series):
            data = GeoSeries(data)
        if data.crs is None and crs is not None:
            data = data.copy()
            data.crs = crs
        return data
    elif isinstance(data, Series):
        out = from_shapely(np.asarray(data), crs=crs)
        return GeoSeries(out, index=data.index, name=data.name)
    else:
        out = from_shapely(data, crs=crs)
        return out