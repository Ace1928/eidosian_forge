from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@staticmethod
def _build_corr_table_nan(sum_of_pairwise_mul: pandas.DataFrame, means: pandas.DataFrame, sums: pandas.DataFrame, count: pandas.DataFrame, std: pandas.DataFrame, cols: pandas.Index, min_periods: int) -> pandas.DataFrame:
    """
        Build correlation matrix for a DataFrame that had NaN values in it.

        Parameters
        ----------
        sum_of_pairwise_mul : pandas.DataFrame
        means : pandas.DataFrame
        sums : pandas.DataFrame
        count : pandas.DataFrame
        std : pandas.DataFrame
        cols : pandas.Index
        min_periods : int

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.
        """
    res = pandas.DataFrame(index=cols, columns=cols, dtype='float')
    nan_mask = count < min_periods
    for col in cols:
        top = sum_of_pairwise_mul.loc[col] - sums.loc[col] * means[col] - means.loc[col] * sums[col] + count.loc[col] * means.loc[col] * means[col]
        down = std.loc[col] * std[col]
        res.loc[col, :] = top / down
    res[nan_mask] = np.nan
    return res