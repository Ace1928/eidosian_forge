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
class category_modulo(category_codes):
    """
    A variation on category_codes that assigns categories using an integer column, modulo a base.
    Category is computed as (column_value - offset)%modulo.
    """
    IntegerTypes = {ct.bool_, ct.uint8, ct.uint16, ct.uint32, ct.uint64, ct.int8, ct.int16, ct.int32, ct.int64}

    def __init__(self, column, modulo, offset=0):
        super().__init__(column)
        self.offset = offset
        self.modulo = modulo

    def _hashable_inputs(self):
        return super()._hashable_inputs() + (self.offset, self.modulo)

    def categories(self, in_dshape):
        return list(range(self.modulo))

    def validate(self, in_dshape):
        if self.column not in in_dshape.dict:
            raise ValueError('specified column not found')
        if in_dshape.measure[self.column] not in self.IntegerTypes:
            raise ValueError('input must be an integer column')

    def apply(self, df, cuda):
        result = (df[self.column] - self.offset) % self.modulo
        if cudf and isinstance(df, cudf.Series):
            if Version(cudf.__version__) >= Version('22.02'):
                return result.to_cupy()
            return result.to_gpu_array()
        else:
            return result.values