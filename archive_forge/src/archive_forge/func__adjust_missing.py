import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _adjust_missing(self):
    """
        Implements alternatives for handling missing values
        """

    def keep_col(x):
        index = np.logical_not(np.any(np.isnan(x), 0))
        return (x[:, index], index)

    def keep_row(x):
        index = np.logical_not(np.any(np.isnan(x), 1))
        return (x[index, :], index)
    if self._missing == 'drop-col':
        self._adjusted_data, index = keep_col(self.data)
        self.cols = np.where(index)[0]
        self.weights = self.weights[index]
    elif self._missing == 'drop-row':
        self._adjusted_data, index = keep_row(self.data)
        self.rows = np.where(index)[0]
    elif self._missing == 'drop-min':
        drop_col, drop_col_index = keep_col(self.data)
        drop_col_size = drop_col.size
        drop_row, drop_row_index = keep_row(self.data)
        drop_row_size = drop_row.size
        if drop_row_size > drop_col_size:
            self._adjusted_data = drop_row
            self.rows = np.where(drop_row_index)[0]
        else:
            self._adjusted_data = drop_col
            self.weights = self.weights[drop_col_index]
            self.cols = np.where(drop_col_index)[0]
    elif self._missing == 'fill-em':
        self._adjusted_data = self._fill_missing_em()
    elif self._missing is None:
        if not np.isfinite(self._adjusted_data).all():
            raise ValueError('data contains non-finite values (inf, NaN). You should drop these values or\nuse one of the methods for adjusting data for missing-values.')
    else:
        raise ValueError('missing method is not known.')
    if self._index is not None:
        self._columns = self._columns[self.cols]
        self._index = self._index[self.rows]
    if self._adjusted_data.size == 0:
        raise ValueError('Removal of missing values has eliminated all data.')