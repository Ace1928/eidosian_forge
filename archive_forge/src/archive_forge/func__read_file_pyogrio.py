import os
from packaging.version import Version
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import pyproj
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from geopandas import GeoDataFrame, GeoSeries
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_netloc, uses_params, uses_relative
import urllib.request
def _read_file_pyogrio(path_or_bytes, bbox=None, mask=None, rows=None, **kwargs):
    import pyogrio
    if rows is not None:
        if isinstance(rows, int):
            kwargs['max_features'] = rows
        elif isinstance(rows, slice):
            if rows.start is not None:
                if rows.start < 0:
                    raise ValueError("Negative slice start not supported with the 'pyogrio' engine.")
                kwargs['skip_features'] = rows.start
            if rows.stop is not None:
                kwargs['max_features'] = rows.stop - (rows.start or 0)
            if rows.step is not None:
                raise ValueError('slice with step is not supported')
        else:
            raise TypeError("'rows' must be an integer or a slice.")
    if bbox is not None:
        if isinstance(bbox, (GeoDataFrame, GeoSeries)):
            bbox = tuple(bbox.total_bounds)
        elif isinstance(bbox, BaseGeometry):
            bbox = bbox.bounds
        if len(bbox) != 4:
            raise ValueError("'bbox' should be a length-4 tuple.")
    if mask is not None:
        raise ValueError("The 'mask' keyword is not supported with the 'pyogrio' engine. You can use 'bbox' instead.")
    if kwargs.pop('ignore_geometry', False):
        kwargs['read_geometry'] = False
    return pyogrio.read_dataframe(path_or_bytes, bbox=bbox, **kwargs)