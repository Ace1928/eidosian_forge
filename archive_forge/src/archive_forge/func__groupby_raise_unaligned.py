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
def _groupby_raise_unaligned(df, convert_by_to_list=True, **kwargs):
    """Groupby, but raise if df and `by` key are unaligned.

    Pandas supports grouping by a column that doesn't align with the input
    frame/series/index. However, the reindexing does not seem to be
    threadsafe, and can result in incorrect results. Since grouping by an
    unaligned key is generally a bad idea, we just error loudly in dask.

    For more information see pandas GH issue #15244 and Dask GH issue #1876."""
    by = kwargs.get('by', None)
    if by is not None and (not _is_aligned(df, by)):
        msg = "Grouping by an unaligned column is unsafe and unsupported.\nThis can be caused by filtering only one of the object or\ngrouping key. For example, the following works in pandas,\nbut not in dask:\n\ndf[df.foo < 0].groupby(df.bar)\n\nThis can be avoided by either filtering beforehand, or\npassing in the name of the column instead:\n\ndf2 = df[df.foo < 0]\ndf2.groupby(df2.bar)\n# or\ndf[df.foo < 0].groupby('bar')\n\nFor more information see dask GH issue #1876."
        raise ValueError(msg)
    elif by is not None and len(by) and convert_by_to_list:
        if isinstance(by, str):
            by = [by]
        kwargs.update(by=list(by))
    with check_observed_deprecation():
        return df.groupby(**kwargs)