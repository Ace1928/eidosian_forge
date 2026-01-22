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
def _compute_lookup(self, row_loc, col_loc):
    """
        Compute index and column labels from index and column integer locators.

        Parameters
        ----------
        row_loc : slice, list, array or tuple
            Row locator.
        col_loc : slice, list, array or tuple
            Columns locator.

        Returns
        -------
        row_lookup : slice(None) if full axis grab, pandas.RangeIndex if repetition is detected, numpy.ndarray otherwise
            List of index labels.
        col_lookup : slice(None) if full axis grab, pandas.RangeIndex if repetition is detected, numpy.ndarray otherwise
            List of columns labels.

        Notes
        -----
        Usage of `slice(None)` as a resulting lookup is a hack to pass information about
        full-axis grab without computing actual indices that triggers lazy computations.
        Ideally, this API should get rid of using slices as indexers and either use a
        common ``Indexer`` object or range and ``np.ndarray`` only.
        """
    lookups = []
    for axis, axis_loc in enumerate((row_loc, col_loc)):
        if is_scalar(axis_loc):
            axis_loc = np.array([axis_loc])
        if isinstance(axis_loc, slice):
            axis_lookup = axis_loc if axis_loc == slice(None) else pandas.RangeIndex(*axis_loc.indices(len(self.arr._query_compiler.get_axis(axis))))
        elif is_range_like(axis_loc):
            axis_lookup = pandas.RangeIndex(axis_loc.start, axis_loc.stop, axis_loc.step)
        elif is_boolean_array(axis_loc):
            axis_lookup = boolean_mask_to_numeric(axis_loc)
        else:
            if isinstance(axis_loc, pandas.Index):
                axis_loc = axis_loc.values
            elif is_list_like(axis_loc) and (not isinstance(axis_loc, np.ndarray)):
                axis_loc = np.array(axis_loc, dtype=np.int64)
            if isinstance(axis_loc, np.ndarray) and (not (axis_loc < 0).any()):
                axis_lookup = axis_loc
            else:
                axis_lookup = pandas.RangeIndex(len(self.arr._query_compiler.get_axis(axis)))[axis_loc]
        if isinstance(axis_lookup, pandas.Index) and (not is_range_like(axis_lookup)):
            axis_lookup = axis_lookup.values
        lookups.append(axis_lookup)
    return lookups