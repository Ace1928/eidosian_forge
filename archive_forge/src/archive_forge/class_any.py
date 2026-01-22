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
class any(OptionalFieldReduction):
    """Whether any elements in ``column`` map to each bin.

    Parameters
    ----------
    column : str, optional
        If provided, any elements in ``column`` that are ``NaN`` are skipped.
    """

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.float32) if antialias else dshape(ct.bool_)

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        return (AntialiasStage2(AntialiasCombination.MAX, array_module.nan),)

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        if not isnull(field):
            agg[y, x] = True
            return 0
        return -1

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor):
        if not isnull(field):
            if isnull(agg[y, x]) or aa_factor > agg[y, x]:
                agg[y, x] = aa_factor
                return 0
        return -1

    @staticmethod
    @ngjit
    def _append_no_field(x, y, agg):
        agg[y, x] = True
        return 0

    @staticmethod
    @ngjit
    def _append_no_field_antialias(x, y, agg, aa_factor, prev_aa_factor):
        if isnull(agg[y, x]) or aa_factor > agg[y, x]:
            agg[y, x] = aa_factor
            return 0
        return -1
    _append_cuda = _append
    _append_no_field_cuda = _append_no_field

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical=False):
        if antialias:
            return self._combine_antialias
        else:
            return self._combine

    @staticmethod
    def _combine(aggs):
        return aggs.sum(axis=0, dtype='bool')

    @staticmethod
    def _combine_antialias(aggs):
        ret = aggs[0]
        for i in range(1, len(aggs)):
            nanmax_in_place(ret, aggs[i])
        return ret