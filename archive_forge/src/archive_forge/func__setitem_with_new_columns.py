from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from .dataframe import DataFrame
from .series import Series
from .utils import is_scalar
def _setitem_with_new_columns(self, row_loc, col_loc, item):
    """
        Assign `item` value to dataset located by `row_loc` and `col_loc` with new columns.

        Parameters
        ----------
        row_loc : scalar, slice, list, array or tuple
            Row locator.
        col_loc : list, array or tuple
            Columns locator.
        item : modin.pandas.DataFrame, modin.pandas.Series or scalar
            Value that should be assigned to located dataset.
        """
    if is_list_like(item) and (not isinstance(item, (DataFrame, Series))):
        item = np.array(item)
        if len(item.shape) == 1:
            if len(col_loc) != 1:
                raise ValueError('Must have equal len keys and value when setting with an iterable')
        elif item.shape[-1] != len(col_loc):
            raise ValueError('Must have equal len keys and value when setting with an iterable')
    common_label_loc = np.isin(col_loc, self.qc.columns.values)
    if not all(common_label_loc):
        columns = self.qc.columns
        for i in range(len(common_label_loc)):
            if not common_label_loc[i]:
                columns = columns.insert(len(columns), col_loc[i])
        self.qc = self.qc.reindex(labels=columns, axis=1, fill_value=np.NaN)
        self.df._update_inplace(new_query_compiler=self.qc)
    self._set_item_existing_loc(row_loc, np.array(col_loc), item)