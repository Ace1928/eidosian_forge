from __future__ import annotations
import pandas as pd
from dask import is_dask_collection
from dask.utils import Dispatch
from_pyarrow_table_dispatch = Dispatch("from_pyarrow_table_dispatch")
def union_categoricals(to_union, sort_categories=False, ignore_order=False):
    func = union_categoricals_dispatch.dispatch(type(to_union[0]))
    return func(to_union, sort_categories=sort_categories, ignore_order=ignore_order)