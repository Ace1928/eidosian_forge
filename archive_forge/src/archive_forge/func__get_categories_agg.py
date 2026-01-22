from __future__ import annotations
from collections import defaultdict
from numbers import Integral
import pandas as pd
from pandas.api.types import is_scalar
from tlz import partition_all
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.accessor import Accessor
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
def _get_categories_agg(parts):
    res = defaultdict(list)
    res_ind = []
    for p in parts:
        for k, v in p[0].items():
            res[k].append(v)
        res_ind.append(p[1])
    res = {k: methods.concat(v, ignore_index=True).drop_duplicates() for k, v in res.items()}
    if res_ind[0] is None:
        return (res, None)
    return (res, res_ind[0].append(res_ind[1:]).drop_duplicates())