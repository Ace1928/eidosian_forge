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
class FloatingNReduction(OptionalFieldReduction):

    def __init__(self, column=None, n=1):
        super().__init__(column)
        self.n = n if n >= 1 else 1

    def out_dshape(self, in_dshape, antialias, cuda, partitioned):
        return dshape(ct.float64)

    def _add_finalize_kwargs(self, **kwargs):
        n_name = 'n'
        n_values = np.arange(self.n)
        kwargs = copy.deepcopy(kwargs)
        kwargs['dims'] += [n_name]
        kwargs['coords'][n_name] = n_values
        return kwargs

    def _build_create(self, required_dshape):
        return lambda shape, array_module: super(FloatingNReduction, self)._build_create(required_dshape)(shape + (self.n,), array_module)

    def _build_finalize(self, dshape):

        def finalize(bases, cuda=False, **kwargs):
            kwargs = self._add_finalize_kwargs(**kwargs)
            return self._finalize(bases, cuda=cuda, **kwargs)
        return finalize

    def _hashable_inputs(self):
        return super()._hashable_inputs() + (self.n,)