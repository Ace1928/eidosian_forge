from __future__ import annotations
from typing import TYPE_CHECKING
from pandas._libs import lib
from pandas.core.dtypes.missing import notna
from pandas.core.algorithms import factorize
from pandas.core.indexes.api import MultiIndex
from pandas.core.series import Series

    Convert a scipy.sparse.coo_matrix to a Series with type sparse.

    Parameters
    ----------
    A : scipy.sparse.coo_matrix
    dense_index : bool, default False

    Returns
    -------
    Series

    Raises
    ------
    TypeError if A is not a coo_matrix
    