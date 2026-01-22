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
class _min_row_index(_max_or_min_row_index):
    """Min reduction operating on row index.

    This is a private class as it is not intended to be used explicitly in
    user code. It is primarily purpose is to support the use of ``first``
    reductions using dask and/or CUDA.
    """

    def _antialias_requires_2_stages(self):
        return True

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MIN, -1),)

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if field != -1 and (agg[y, x] == -1 or field < agg[y, x]):
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        if field != -1 and (agg[y, x] == -1 or field < agg[y, x]):
            agg[y, x] = field
            return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        if field != -1 and (agg[y, x] == -1 or field < agg[y, x]):
            agg[y, x] = field
            return 0
        return -1

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical=False):
        if cuda:
            return self._combine_cuda
        else:
            return self._combine

    @staticmethod
    def _combine(aggs):
        ret = aggs[0]
        for i in range(1, len(aggs)):
            row_min_in_place(ret, aggs[i])
        return ret

    @staticmethod
    def _combine_cuda(aggs):
        ret = aggs[0]
        if len(aggs) > 1:
            if ret.ndim == 2:
                aggs = [cp.expand_dims(agg, 2) for agg in aggs]
            kernel_args = cuda_args(ret.shape[:3])
            for i in range(1, len(aggs)):
                cuda_row_min_in_place[kernel_args](aggs[0], aggs[i])
        return ret