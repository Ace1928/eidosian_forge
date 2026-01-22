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
def _parse_row_and_column_locators(self, tup):
    """
        Unpack the user input for getitem and setitem and compute ndim.

        loc[a] -> ([a], :), 1D
        loc[[a,b]] -> ([a,b], :),
        loc[a,b] -> ([a], [b]), 0D

        Parameters
        ----------
        tup : tuple
            User input to unpack.

        Returns
        -------
        row_loc : scalar or list
            Row locator(s) as a scalar or List.
        col_list : scalar or list
            Column locator(s) as a scalar or List.
        ndim : {0, 1, 2}
            Number of dimensions of located dataset.
        """
    row_loc, col_loc = (slice(None), slice(None))
    if is_tuple(tup):
        row_loc = tup[0]
        if len(tup) == 2:
            col_loc = tup[1]
        if len(tup) > 2:
            raise IndexingError('Too many indexers')
    else:
        row_loc = tup
    row_loc = row_loc(self.arr) if callable(row_loc) else row_loc
    col_loc = col_loc(self.arr) if callable(col_loc) else col_loc
    row_loc = row_loc._to_numpy() if isinstance(row_loc, array) else row_loc
    col_loc = col_loc._to_numpy() if isinstance(col_loc, array) else col_loc
    return (row_loc, col_loc, _compute_ndim(row_loc, col_loc))