from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import algos as libalgos
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import BaseMaskedDtype
class SelectNFrame(SelectN):
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
    n : int
    keep : {'first', 'last'}, default 'first'
    columns : list or str

    Returns
    -------
    nordered : DataFrame
    """

    def __init__(self, obj: DataFrame, n: int, keep: str, columns: IndexLabel) -> None:
        super().__init__(obj, n, keep)
        if not is_list_like(columns) or isinstance(columns, tuple):
            columns = [columns]
        columns = cast(Sequence[Hashable], columns)
        columns = list(columns)
        self.columns = columns

    def compute(self, method: str) -> DataFrame:
        from pandas.core.api import Index
        n = self.n
        frame = self.obj
        columns = self.columns
        for column in columns:
            dtype = frame[column].dtype
            if not self.is_valid_dtype_n_method(dtype):
                raise TypeError(f'Column {repr(column)} has dtype {dtype}, cannot use method {repr(method)} with this dtype')

        def get_indexer(current_indexer, other_indexer):
            """
            Helper function to concat `current_indexer` and `other_indexer`
            depending on `method`
            """
            if method == 'nsmallest':
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer = Index([], dtype=np.int64)
        for i, column in enumerate(columns):
            series = cur_frame[column]
            is_last_column = len(columns) - 1 == i
            values = getattr(series, method)(cur_n, keep=self.keep if is_last_column else 'all')
            if is_last_column or len(values) <= cur_n:
                indexer = get_indexer(indexer, values.index)
                break
            border_value = values == values[values.index[-1]]
            unsafe_values = values[border_value]
            safe_values = values[~border_value]
            indexer = get_indexer(indexer, safe_values.index)
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = n - len(indexer)
        frame = frame.take(indexer)
        frame.index = original_index.take(indexer)
        if len(columns) == 1:
            return frame
        ascending = method == 'nsmallest'
        return frame.sort_values(columns, ascending=ascending, kind='mergesort')