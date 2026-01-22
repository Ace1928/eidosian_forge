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
class _upsample(Reduction):
    """"Special internal class used for upsampling"""

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(Option(ct.float64))

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[0], **kwargs)

    @property
    def inputs(self):
        return (extract(self.column),)

    def _build_create(self, required_dshape):
        return self._create_float64_empty

    @staticmethod
    @ngjit
    def _append(x, y, agg, field):
        pass

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field):
        pass

    @staticmethod
    def _combine(aggs):
        return np.nanmax(aggs, axis=0)