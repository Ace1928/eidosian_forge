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
class _max_row_index(_max_or_min_row_index):
    """Max reduction operating on row index.

    This is a private class as it is not intended to be used explicitly in
    user code. It is primarily purpose is to support the use of ``last``
    reductions using dask and/or CUDA.
    """

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MAX, -1),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if field > agg[y, x]:
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        if field > agg[y, x]:
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if field != -1:
            old = nb_cuda.atomic.max(agg, (y, x), field)
            if old < field:
                return 0
        return -1

    @staticmethod
    def _combine(aggs):
        ret = aggs[0]
        for i in range(1, len(aggs)):
            np.maximum(ret, aggs[i], out=ret)
        return ret