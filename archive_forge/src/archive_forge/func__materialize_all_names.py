from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def _materialize_all_names(self):
    """Materialize missing column names."""
    if self._know_all_names:
        return
    all_cols = self._parent_df.columns
    self._normalize_self_levels(all_cols)
    for col in all_cols:
        if col not in self._known_dtypes and col not in self._cols_with_unknown_dtypes:
            self._cols_with_unknown_dtypes.append(col)
    self._know_all_names = True