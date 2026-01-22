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
class SpecialColumn(Enum):
    """
    Internally datashader identifies the columns required by the user's
    Reductions and extracts them from the supplied source (e.g. DataFrame) to
    pass through the dynamically-generated append function in compiler.py and
    end up as arguments to the Reduction._append* functions. Each column is
    a string name or a SpecialColumn. A column of None is used in Reduction
    classes to denote that no column is required.
    """
    RowIndex = 1