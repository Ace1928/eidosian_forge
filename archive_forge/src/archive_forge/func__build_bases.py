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
def _build_bases(self, cuda, partitioned):
    selector = self.selector
    if isinstance(selector, (_first_or_last, _first_n_or_last_n)) and selector.uses_row_index(cuda, partitioned):
        row_index_selector = selector._create_row_index_selector()
        if self.column == SpecialColumn.RowIndex:
            row_index_selector._nan_check_column = self.selector.column
            return row_index_selector._build_bases(cuda, partitioned)
        else:
            new_where = where(row_index_selector, self.column)
            new_where._nan_check_column = self.selector.column
            return row_index_selector._build_bases(cuda, partitioned) + new_where._build_bases(cuda, partitioned)
    else:
        return selector._build_bases(cuda, partitioned) + super()._build_bases(cuda, partitioned)