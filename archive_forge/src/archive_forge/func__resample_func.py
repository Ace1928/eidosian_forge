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
def _resample_func(self, resample_kwargs, func_name, new_columns=None, df_op=None, allow_range_impl=True, *args, **kwargs):
    """
        Resample underlying time-series data and apply aggregation on it.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters in the format of ``modin.pandas.DataFrame.resample`` signature.
        func_name : str
            Aggregation function name to apply on resampler object.
        new_columns : list of labels, optional
            Actual column labels of the resulted frame, supposed to be a hint for the
            Modin frame. If not specified will be computed automaticly.
        df_op : callable(pandas.DataFrame) -> [pandas.DataFrame, pandas.Series], optional
            Preprocessor function to apply to the passed frame before resampling.
        allow_range_impl : bool, default: True
            Whether to use range-partitioning if ``RangePartitioning.get() is True``.
        *args : args
            Arguments to pass to the aggregation function.
        **kwargs : kwargs
            Arguments to pass to the aggregation function.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the result of resample aggregation.
        """
    from modin.core.dataframe.pandas.dataframe.utils import ShuffleResample

    def map_func(df, resample_kwargs=resample_kwargs):
        """Resample time-series data of the passed frame and apply aggregation function on it."""
        if len(df) == 0:
            if resample_kwargs['on'] is not None:
                df = df.set_index(resample_kwargs['on'])
            return df
        if 'bin_bounds' in df.attrs:
            timestamps = df.attrs['bin_bounds']
            if isinstance(df.index, pandas.MultiIndex):
                level_to_keep = resample_kwargs['level']
                if isinstance(level_to_keep, int):
                    to_drop = [lvl for lvl in range(df.index.nlevels) if lvl != level_to_keep]
                else:
                    to_drop = [lvl for lvl in df.index.names if lvl != level_to_keep]
                df.index = df.index.droplevel(to_drop)
                resample_kwargs = resample_kwargs.copy()
                resample_kwargs['level'] = None
            filler = pandas.DataFrame(np.NaN, index=pandas.Index(timestamps), columns=df.columns)
            df = pandas.concat([df, filler], copy=False)
        if df_op is not None:
            df = df_op(df)
        resampled_val = df.resample(**resample_kwargs)
        op = getattr(pandas.core.resample.Resampler, func_name)
        if callable(op):
            try:
                val = op(resampled_val, *args, **kwargs)
            except ValueError:
                resampled_val = df.copy().resample(**resample_kwargs)
                val = op(resampled_val, *args, **kwargs)
        else:
            val = getattr(resampled_val, func_name)
        if isinstance(val, pandas.Series):
            return val.to_frame()
        else:
            return val
    if resample_kwargs['on'] is None:
        level = [0 if resample_kwargs['level'] is None else resample_kwargs['level']]
        key_columns = []
    else:
        level = None
        key_columns = [resample_kwargs['on']]
    if not allow_range_impl or resample_kwargs['axis'] not in (0, 'index') or (not RangePartitioning.get()):
        new_modin_frame = self._modin_frame.apply_full_axis(axis=0, func=map_func, new_columns=new_columns)
    else:
        new_modin_frame = self._modin_frame._apply_func_to_range_partitioning(key_columns=key_columns, level=level, func=map_func, shuffle_func_cls=ShuffleResample, resample_kwargs=resample_kwargs)
    return self.__constructor__(new_modin_frame)