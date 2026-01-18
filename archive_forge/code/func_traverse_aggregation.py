from __future__ import annotations
from itertools import count
import logging
from typing import TYPE_CHECKING
from toolz import unique, concat, pluck, get, memoize
from numba import literal_unroll
import numpy as np
import xarray as xr
from .antialias import AntialiasCombination
from .reductions import SpecialColumn, UsesCudaMutex, by, category_codes, summary
from .utils import (isnull, ngjit,
def traverse_aggregation(agg):
    """Yield a left->right traversal of an aggregation"""
    if isinstance(agg, summary):
        for a in agg.values:
            for a2 in traverse_aggregation(a):
                yield a2
    else:
        yield agg