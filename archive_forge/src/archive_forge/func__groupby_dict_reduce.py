import ast
import hashlib
import re
import warnings
from collections.abc import Iterable
from typing import Hashable, List
import numpy as np
import pandas
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.groupby.base import transformation_kernels
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.errors import DataError
from modin.config import CpuCount, RangePartitioning, use_range_partitioning_groupby
from modin.core.dataframe.algebra import (
from modin.core.dataframe.algebra.default2pandas.groupby import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import get_logger
from modin.utils import (
from .aggregations import CorrCovBuilder
from .groupby import GroupbyReduceImpl, PivotTableImpl
from .merge import MergeImpl
from .utils import get_group_names, merge_partitioning
def _groupby_dict_reduce(self, by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False, **kwargs):
    """
        Group underlying data and apply aggregation functions to each group of the specified column/row.

        This method is responsible of performing dictionary groupby aggregation for such functions,
        that can be implemented via TreeReduce approach.

        Parameters
        ----------
        by : PandasQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        agg_func : dict(label) -> str
            Dictionary that maps row/column labels to the function names.
            **Note:** specified functions have to be supported by ``modin.core.dataframe.algebra.GroupByReduce``.
            Supported functions are listed in the ``modin.core.dataframe.algebra.GroupByReduce.groupby_reduce_functions``
            dictionary.
        axis : {0, 1}
            Axis to group and apply aggregation function along.
            0 is for index, when 1 is for columns.
        groupby_kwargs : dict
            GroupBy parameters in the format of ``modin.pandas.DataFrame.groupby`` signature.
        agg_args : list-like
            Serves the compatibility purpose. Does not affect the result.
        agg_kwargs : dict
            Arguments to pass to the aggregation functions.
        drop : bool, default: False
            If `by` is a QueryCompiler indicates whether or not by-data came
            from the `self`.
        **kwargs : dict
            Additional parameters to pass to the ``modin.core.dataframe.algebra.GroupByReduce.register``.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the result of groupby dictionary aggregation.
        """
    map_dict = {}
    reduce_dict = {}
    kwargs.setdefault('default_to_pandas_func', lambda grp, *args, **kwargs: grp.agg(agg_func, *args, **kwargs))
    rename_columns = any((not isinstance(fn, str) and isinstance(fn, Iterable) for fn in agg_func.values()))
    for col, col_funcs in agg_func.items():
        if not rename_columns:
            map_dict[col], reduce_dict[col], _ = GroupbyReduceImpl.get_impl(col_funcs)
            continue
        if isinstance(col_funcs, str):
            col_funcs = [col_funcs]
        map_fns = []
        for i, fn in enumerate(col_funcs):
            if not isinstance(fn, str) and isinstance(fn, Iterable):
                new_col_name, func = fn
            elif isinstance(fn, str):
                new_col_name, func = (fn, fn)
            else:
                raise TypeError
            map_fn, reduce_fn, _ = GroupbyReduceImpl.get_impl(func)
            map_fns.append((new_col_name, map_fn))
            reduced_col_name = (*col, new_col_name) if isinstance(col, tuple) else (col, new_col_name)
            reduce_dict[reduced_col_name] = reduce_fn
        map_dict[col] = map_fns
    return GroupByReduce.register(map_dict, reduce_dict, **kwargs)(query_compiler=self, by=by, axis=axis, groupby_kwargs=groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop)