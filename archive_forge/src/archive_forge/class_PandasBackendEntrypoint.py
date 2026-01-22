from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, union_categoricals
from dask.array.core import Array
from dask.array.dispatch import percentile_lookup
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
from dask.dataframe._compat import PANDAS_GE_220, is_any_real_numeric_dtype
from dask.dataframe.core import DataFrame, Index, Scalar, Series, _Frame
from dask.dataframe.dispatch import (
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.dataframe.utils import (
from dask.sizeof import SimpleSizeof, sizeof
from dask.utils import is_arraylike, is_series_like, typename
class PandasBackendEntrypoint(DataFrameBackendEntrypoint):
    """Pandas-Backend Entrypoint Class for Dask-DataFrame

    Note that all DataFrame-creation functions are defined
    and registered 'in-place' within the ``dask.dataframe``
    ``io`` module.
    """

    @classmethod
    def to_backend_dispatch(cls):
        return to_pandas_dispatch

    @classmethod
    def to_backend(cls, data: _Frame, **kwargs):
        if isinstance(data._meta, (pd.DataFrame, pd.Series, pd.Index)):
            return data
        return data.map_partitions(cls.to_backend_dispatch(), **kwargs)