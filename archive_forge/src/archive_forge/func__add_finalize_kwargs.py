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
def _add_finalize_kwargs(self, **kwargs):
    n_name = 'n'
    n_values = np.arange(self.n)
    kwargs = copy.deepcopy(kwargs)
    kwargs['dims'] += [n_name]
    kwargs['coords'][n_name] = n_values
    return kwargs