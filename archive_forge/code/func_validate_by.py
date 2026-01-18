import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@classmethod
def validate_by(cls, by):
    """
        Build valid `by` parameter for `pandas.DataFrame.groupby`.

        Cast all DataFrames in `by` parameter to Series or list of Series in case
        of multi-column frame.

        Parameters
        ----------
        by : DateFrame, Series, index label or list of such
            Object which indicates groups for GroupBy.

        Returns
        -------
        Series, index label or list of such
            By parameter with all DataFrames casted to Series.
        """

    def try_cast_series(df):
        """Cast one-column frame to Series."""
        if isinstance(df, pandas.DataFrame):
            df = df.squeeze(axis=1)
        if not isinstance(df, pandas.Series):
            return df
        if df.name == MODIN_UNNAMED_SERIES_LABEL:
            df.name = None
        return df
    if isinstance(by, pandas.DataFrame):
        by = [try_cast_series(column) for _, column in by.items()]
    elif isinstance(by, pandas.Series):
        by = [try_cast_series(by)]
    elif isinstance(by, list):
        by = [try_cast_series(o) for o in by]
    return by