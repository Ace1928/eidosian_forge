import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def _make_df_square(table):
    """
    Reindex a pandas DataFrame so that it becomes square, meaning that
    the row and column indices contain the same values, in the same
    order.  The row and column index are extended to achieve this.
    """
    if not isinstance(table, pd.DataFrame):
        return table
    if not table.index.equals(table.columns):
        ix = list(set(table.index) | set(table.columns))
        ix.sort()
        table = table.reindex(index=ix, columns=ix, fill_value=0)
    table = table.reindex(table.columns)
    return table