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
def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
    if cuda:
        if antialias:
            return self._append_antialias_cuda
        else:
            return self._append_cuda
    elif antialias:
        return self._append_antialias
    else:
        return self._append