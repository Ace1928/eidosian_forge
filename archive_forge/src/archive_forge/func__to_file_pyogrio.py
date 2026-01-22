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
def _to_file_pyogrio(df, filename, driver, schema, crs, mode, **kwargs):
    import pyogrio
    if schema is not None:
        raise ValueError("The 'schema' argument is not supported with the 'pyogrio' engine.")
    if mode == 'a':
        kwargs['append'] = True
    if crs is not None:
        raise ValueError("Passing 'crs' it not supported with the 'pyogrio' engine.")
    if not df.columns.is_unique:
        raise ValueError('GeoDataFrame cannot contain duplicated column names.')
    pyogrio.write_dataframe(df, filename, driver=driver, **kwargs)