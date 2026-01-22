from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
@property
def _meta_nonempty(self):
    """
        Return a pd.DataFrameGroupBy / pd.SeriesGroupBy which contains sample data.
        """
    sample = self.obj._meta_nonempty
    if isinstance(self.by, list):
        by_meta = [item._meta_nonempty if isinstance(item, Series) else item for item in self.by]
    elif isinstance(self.by, Series):
        by_meta = self.by._meta_nonempty
    else:
        by_meta = self.by
    with check_observed_deprecation():
        grouped = sample.groupby(by_meta, group_keys=self.group_keys, **self.observed, **self.dropna)
    return _maybe_slice(grouped, self._slice)