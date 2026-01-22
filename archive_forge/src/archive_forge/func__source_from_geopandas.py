from __future__ import annotations
from numbers import Number
from math import log10
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from xarray import DataArray, Dataset
from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, \
from .utils import get_indices, dshape_from_pandas, dshape_from_dask
from .utils import Expr # noqa (API import)
from .resampling import resample_2d, resample_2d_distributed
from . import reductions as rd
def _source_from_geopandas(self, source):
    """
        Check if the specified source is a geopandas or dask-geopandas GeoDataFrame.
        If so, spatially filter the source and return it.
        If not, return None.
        """
    try:
        import geopandas
    except ImportError:
        geopandas = None
    try:
        import dask_geopandas
    except ImportError:
        dask_geopandas = None
    if geopandas and isinstance(source, geopandas.GeoDataFrame) or (dask_geopandas and isinstance(source, dask_geopandas.GeoDataFrame)):
        from packaging.version import Version
        from shapely import __version__ as shapely_version
        if Version(shapely_version) < Version('2.0.0'):
            raise ImportError('Use of GeoPandas in Datashader requires Shapely >= 2.0.0')
        if isinstance(source, geopandas.GeoDataFrame):
            x_range = self.x_range if self.x_range is not None else (-np.inf, np.inf)
            y_range = self.y_range if self.y_range is not None else (-np.inf, np.inf)
            from shapely import box
            query = source.sindex.query(box(x_range[0], y_range[0], x_range[1], y_range[1]))
            source = source.iloc[query]
        else:
            x_range = self.x_range if self.x_range is not None else (None, None)
            y_range = self.y_range if self.y_range is not None else (None, None)
            source = source.cx[slice(*x_range), slice(*y_range)]
        return source
    else:
        return None