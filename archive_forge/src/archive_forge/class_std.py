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
class std(Reduction):
    """Standard Deviation of all elements in ``column``.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """

    def _build_bases(self, cuda, partitioned):
        return (_sum_zero(self.column), count(self.column), m2(self.column))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        sums, counts, m2s = bases
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(counts > 0, np.sqrt(m2s / counts), np.nan)
        return xr.DataArray(x, **kwargs)