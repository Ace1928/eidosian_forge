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
def _set_item_existing_loc(self, row_loc, col_loc, item):
    """
        Assign `item` value to dataset located by `row_loc` and `col_loc` with existing rows and columns.

        Parameters
        ----------
        row_loc : scalar, slice, list, array or tuple
            Row locator.
        col_loc : scalar, slice, list, array or tuple
            Columns locator.
        item : modin.pandas.DataFrame, modin.pandas.Series or scalar
            Value that should be assigned to located dataset.
        """
    if isinstance(row_loc, Series) and is_boolean_array(row_loc) and is_scalar(item):
        new_qc = self.df._query_compiler.setitem_bool(row_loc._query_compiler, col_loc, item)
        self.df._update_inplace(new_qc)
        return
    row_lookup, col_lookup = self.qc.get_positions_from_labels(row_loc, col_loc)
    if isinstance(item, np.ndarray) and is_boolean_array(row_loc):
        item = item.take(row_lookup)
    self._setitem_positional(row_lookup, col_lookup, item, axis=self._determine_setitem_axis(row_lookup, col_lookup, is_scalar(row_loc), is_scalar(col_loc)))