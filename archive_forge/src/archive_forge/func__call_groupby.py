import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@classmethod
def _call_groupby(cls, df, *args, **kwargs):
    """Call .groupby() on passed `df` squeezed to Series."""
    if len(df.columns) == 1:
        return df.squeeze(axis=1).groupby(*args, **kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        return df.groupby(*args, **kwargs)[df.columns[0]]