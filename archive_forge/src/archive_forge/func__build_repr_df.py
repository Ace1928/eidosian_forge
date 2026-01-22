from __future__ import annotations
import pickle as pkl
import re
import warnings
from typing import TYPE_CHECKING, Any, Hashable, Literal, Optional, Sequence, Union
import numpy as np
import pandas
import pandas.core.generic
import pandas.core.resample
import pandas.core.window.rolling
from pandas._libs import lib
from pandas._libs.tslibs import to_offset
from pandas._typing import (
from pandas.compat import numpy as numpy_compat
from pandas.core.common import count_not_none, pipe
from pandas.core.dtypes.common import (
from pandas.core.indexes.api import ensure_index
from pandas.core.methods.describe import _refine_percentiles
from pandas.util._validators import (
from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.accessor import CachedAccessor, ModinAPI
from modin.pandas.utils import is_scalar
from modin.utils import _inherit_docstrings, expanduser_path_arg, try_cast_to_pandas
from .utils import _doc_binary_op, is_full_grab_slice
def _build_repr_df(self, num_rows, num_cols):
    """
        Build pandas DataFrame for string representation.

        Parameters
        ----------
        num_rows : int
            Number of rows to show in string representation. If number of
            rows in this dataset is greater than `num_rows` then half of
            `num_rows` rows from the beginning and half of `num_rows` rows
            from the end are shown.
        num_cols : int
            Number of columns to show in string representation. If number of
            columns in this dataset is greater than `num_cols` then half of
            `num_cols` columns from the beginning and half of `num_cols`
            columns from the end are shown.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            A pandas dataset with `num_rows` or fewer rows and `num_cols` or fewer columns.
        """
    if len(self.index) == 0 or (self._is_dataframe and len(self.columns) == 0):
        return pandas.DataFrame(index=self.index, columns=self.columns if self._is_dataframe else None)
    row_indexer = _get_repr_axis_label_indexer(self.index, num_rows)
    if self._is_dataframe:
        indexer = (row_indexer, _get_repr_axis_label_indexer(self.columns, num_cols))
    else:
        indexer = row_indexer
    return self.iloc[indexer]._query_compiler.to_pandas()