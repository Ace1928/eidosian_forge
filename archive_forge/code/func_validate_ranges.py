from __future__ import annotations
from numbers import Number
from math import log10
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from xarray import DataArray, Dataset
from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, \
from .utils import get_indices, dshape_from_pandas, dshape_from_dask
from .utils import Expr # noqa (API import)
from .resampling import resample_2d, resample_2d_distributed
from . import reductions as rd
def validate_ranges(self, x_range, y_range):
    self.x_axis.validate(x_range)
    self.y_axis.validate(y_range)