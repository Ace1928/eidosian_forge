import itertools
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.pandas.indexing import compute_sliced_len, is_range_like, is_slice, is_tuple
from modin.pandas.utils import is_scalar
from .arr import array
def _setitem_positional(self, row_lookup, col_lookup, item, axis=None):
    """
        Assign `item` value to located dataset.

        Parameters
        ----------
        row_lookup : slice or scalar
            The global row index to write item to.
        col_lookup : slice or scalar
            The global col index to write item to.
        item : DataFrame, Series or scalar
            The new item needs to be set. It can be any shape that's
            broadcast-able to the product of the lookup tables.
        axis : {None, 0, 1}, default: None
            If not None, it means that whole axis is used to assign a value.
            0 means assign to whole column, 1 means assign to whole row.
            If None, it means that partial assignment is done on both axes.
        """
    if isinstance(row_lookup, slice):
        row_lookup = range(len(self.arr._query_compiler.index))[row_lookup]
    if isinstance(col_lookup, slice):
        col_lookup = range(len(self.arr._query_compiler.columns))[col_lookup]
    new_qc = self.arr._query_compiler.write_items(row_lookup, col_lookup, item)
    self.arr._update_inplace(new_qc)