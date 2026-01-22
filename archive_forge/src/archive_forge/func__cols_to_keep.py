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
def _cols_to_keep(columns, glyph, agg):
    """
    Return which columns from the supplied data source are kept as they are
    needed by the specified agg. Excludes any SpecialColumn.
    """
    cols_to_keep = dict({col: False for col in columns})
    for col in glyph.required_columns():
        cols_to_keep[col] = True

    def recurse(cols_to_keep, agg):
        if hasattr(agg, 'values'):
            for subagg in agg.values:
                recurse(cols_to_keep, subagg)
        elif hasattr(agg, 'columns'):
            for column in agg.columns:
                if column not in (None, rd.SpecialColumn.RowIndex):
                    cols_to_keep[column] = True
        elif agg.column not in (None, rd.SpecialColumn.RowIndex):
            cols_to_keep[agg.column] = True
    recurse(cols_to_keep, agg)
    return [col for col, keepit in cols_to_keep.items() if keepit]