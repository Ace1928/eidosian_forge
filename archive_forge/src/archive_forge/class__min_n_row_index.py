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
class _min_n_row_index(_max_n_or_min_n_row_index):
    """Min_n reduction operating on row index.

    This is a private class as it is not intended to be used explicitly in
    user code. It is primarily purpose is to support the use of ``first_n``
    reductions using dask and/or CUDA.
    """

    def _antialias_requires_2_stages(self):
        return True

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MIN, -1, n_reduction=True),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if field != -1:
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field < agg[y, x, i]:
                    shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        if field != -1:
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field < agg[y, x, i]:
                    shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if field != -1:
            n = agg.shape[2]
            for i in range(n):
                if agg[y, x, i] == -1 or field < agg[y, x, i]:
                    cuda_shift_and_insert(agg[y, x], field, i)
                    return i
        return -1

    @staticmethod
    def _combine(aggs):
        ret = aggs[0]
        if len(aggs) > 1:
            if ret.ndim == 3:
                row_min_n_in_place_3d(aggs[0], aggs[1])
            else:
                row_min_n_in_place_4d(aggs[0], aggs[1])
        return ret

    @staticmethod
    def _combine_cuda(aggs):
        ret = aggs[0]
        if len(aggs) > 1:
            kernel_args = cuda_args(ret.shape[:-1])
            if ret.ndim == 3:
                cuda_row_min_n_in_place_3d[kernel_args](aggs[0], aggs[1])
            else:
                cuda_row_min_n_in_place_4d[kernel_args](aggs[0], aggs[1])
        return ret