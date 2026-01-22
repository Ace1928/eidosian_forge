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
class m2(FloatingReduction):
    """Sum of square differences from the mean of all elements in ``column``.

    Intermediate value for computing ``var`` and ``std``, not intended to be
    used on its own.

    Parameters
    ----------
    column : str
        Name of the column to aggregate over. Column data type must be numeric.
        ``NaN`` values in the column are skipped.
    """

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Global

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        return super(m2, self)._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _build_create(self, required_dshape):
        return self._create_float64_zero

    def _build_temps(self, cuda=False):
        return (_sum_zero(self.column), count(self.column))

    @staticmethod
    @ngjit
    def _append(x, y, m2, field, sum, count):
        if not isnull(field):
            if count > 0:
                u1 = np.float64(sum) / count
                u = np.float64(sum + field) / (count + 1)
                m2[y, x] += (field - u1) * (field - u)
                return 0
        return -1

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, m2, field, sum, count):
        if not isnull(field):
            if count > 0:
                u1 = np.float64(sum) / count
                u = np.float64(sum + field) / (count + 1)
                m2[y, x] += (field - u1) * (field - u)
                return 0
        return -1

    @staticmethod
    def _combine(Ms, sums, ns):
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = np.nansum(sums, axis=0) / ns.sum(axis=0)
            return np.nansum(Ms + ns * (sums / ns - mu) ** 2, axis=0)