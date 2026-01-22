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
@staticmethod
@nb_cuda.jit(device=True)
def _append_no_field_antialias_cuda(x, y, agg, aa_factor, prev_aa_factor):
    if not isnull(aa_factor):
        old = cuda_atomic_nanmax(agg, (y, x), aa_factor)
        if isnull(old) or old < aa_factor:
            return 0
    return -1