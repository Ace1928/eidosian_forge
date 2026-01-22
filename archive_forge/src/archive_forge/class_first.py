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
class first(_first_or_last):
    """First value encountered in ``column``.

    Useful for categorical data where an actual value must always be returned,
    not an average or other numerical calculation.

    Currently only supported for rasters, externally to this class.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. If the data type is floating point,
        ``NaN`` values in the column are skipped.
    """

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.FIRST, array_module.nan),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field) and isnull(agg[y, x]):
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        value = field * aa_factor
        if not isnull(value) and (isnull(agg[y, x]) or value > agg[y, x]):
            agg[y, x] = value
            return 0
        return -1

    def _create_row_index_selector(self):
        return _min_row_index()