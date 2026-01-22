from __future__ import annotations
import copy
from enum import Enum
from packaging.version import Version
import numpy as np
from datashader.datashape import dshape, isnumeric, Record, Option
from datashader.datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr
from datashader.antialias import AntialiasCombination, AntialiasStage2
from datashader.utils import isminus1, isnull
from numba import cuda as nb_cuda
from .utils import (
class extract(Preprocess):
    """Extract a column from a dataframe as a numpy array of values."""

    def apply(self, df, cuda):
        if self.column is SpecialColumn.RowIndex:
            attr_name = '_datashader_row_offset'
            if isinstance(df, xr.Dataset):
                row_offset = df.attrs[attr_name]
                row_length = df.attrs['_datashader_row_length']
            else:
                attrs = getattr(df, 'attrs', None)
                row_offset = getattr(attrs or df, attr_name, 0)
                row_length = len(df)
        if cudf and isinstance(df, cudf.DataFrame):
            if self.column is SpecialColumn.RowIndex:
                return cp.arange(row_offset, row_offset + row_length, dtype=np.int64)
            if df[self.column].dtype.kind == 'f':
                nullval = np.nan
            else:
                nullval = 0
            if Version(cudf.__version__) >= Version('22.02'):
                return df[self.column].to_cupy(na_value=nullval)
            return cp.array(df[self.column].to_gpu_array(fillna=nullval))
        elif self.column is SpecialColumn.RowIndex:
            if cuda:
                return cp.arange(row_offset, row_offset + row_length, dtype=np.int64)
            else:
                return np.arange(row_offset, row_offset + row_length, dtype=np.int64)
        elif isinstance(df, xr.Dataset):
            if cuda and (not isinstance(df[self.column].data, cp.ndarray)):
                return cp.asarray(df[self.column])
            else:
                return df[self.column].data
        else:
            return df[self.column].values