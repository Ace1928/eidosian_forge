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
@ngjit
def _append_antialias_not_self_intersect(x, y, agg, field, aa_factor, prev_aa_factor):
    value = field * aa_factor
    if not isnull(value):
        if isnull(agg[y, x]) or value > agg[y, x]:
            agg[y, x] = value
            return 0
    return -1