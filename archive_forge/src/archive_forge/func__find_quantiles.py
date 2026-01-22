import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._libs.tslibs import to_offset
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.resample import _get_timestamp_range_edges
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
@staticmethod
def _find_quantiles(df: Union[pandas.DataFrame, pandas.Series], quantiles: list, method: str) -> np.ndarray:
    """
        Find quantiles of a given dataframe using the specified method.

        We use this method to provide backwards compatibility with NumPy versions < 1.23 (e.g. when
        the user is using Modin in compat mode). This is basically a wrapper around `np.quantile` that
        ensures we provide the correct `method` argument - i.e. if we are dealing with objects (which
        may or may not support algebra), we do not want to use a method to find quantiles that will
        involve algebra operations (e.g. mean) between the objects, since that may fail.

        Parameters
        ----------
        df : pandas.DataFrame or pandas.Series
            The data to pick quantiles from.
        quantiles : list[float]
            The quantiles to compute.
        method : str
            The method to use. `linear` if dealing with numeric types, otherwise `inverted_cdf`.

        Returns
        -------
        np.ndarray
            A NumPy array with the quantiles of the data.
        """
    if method == 'linear':
        return np.unique(np.quantile(df, quantiles))
    else:
        try:
            return np.unique(np.quantile(df, quantiles, method=method))
        except Exception:
            return np.unique(np.quantile(df, quantiles, interpolation='lower'))