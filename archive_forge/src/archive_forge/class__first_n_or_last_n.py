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
class _first_n_or_last_n(FloatingNReduction):
    """Abstract base class of first_n and last_n reductions.
    """

    def uses_row_index(self, cuda, partitioned):
        return cuda or partitioned

    def _antialias_requires_2_stages(self):
        return True

    def _build_bases(self, cuda, partitioned):
        if self.uses_row_index(cuda, partitioned):
            row_index_selector = self._create_row_index_selector()
            wrapper = where(selector=row_index_selector, lookup_column=self.column)
            wrapper._nan_check_column = self.column
            return row_index_selector._build_bases(cuda, partitioned) + (wrapper,)
        else:
            return super()._build_bases(cuda, partitioned)

    @staticmethod
    def _combine(aggs):
        if len(aggs) > 1:
            raise RuntimeError('_combine should never be called with more than one agg')
        return aggs[0]

    def _create_row_index_selector(self):
        pass

    @staticmethod
    def _finalize(bases, cuda=False, **kwargs):
        return xr.DataArray(bases[-1], **kwargs)