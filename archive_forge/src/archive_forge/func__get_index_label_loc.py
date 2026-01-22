from __future__ import annotations
from statsmodels.compat.pandas import (
import numbers
import warnings
import numpy as np
from pandas import (
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
def _get_index_label_loc(self, key, base_index=None):
    """
        Get the location of a specific key in an index or model row labels

        Parameters
        ----------
        key : label
            The key for which to find the location if the underlying index is
            a DateIndex or is only being used as row labels, or a location if
            the underlying index is a RangeIndex or an NumericIndex.
        base_index : pd.Index, optional
            Optionally the base index to search. If None, the model's index is
            searched.

        Returns
        -------
        loc : int
            The location of the key
        index : pd.Index
            The index including the key; this is a copy of the original index
            unless the index had to be expanded to accommodate `key`.
        index_was_expanded : bool
            Whether or not the index was expanded to accommodate `key`.

        Notes
        -----
        This method expands on `_get_index_loc` by first trying the given
        base index (or the model's index if the base index was not given) and
        then falling back to try again with the model row labels as the base
        index.
        """
    if base_index is None:
        base_index = self._index
    return get_index_label_loc(key, base_index, self.data.row_labels)