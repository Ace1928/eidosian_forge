from __future__ import annotations
import pickle
import warnings
from typing import TYPE_CHECKING, Union
import pandas
from pandas._typing import CompressionOptions, StorageOptions
from pandas.core.dtypes.dtypes import SparseDtype
from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.io import to_dask, to_ray
from modin.utils import _inherit_docstrings
def to_coo(self, row_levels=(0,), column_levels=(1,), sort_labels=False):
    return self._default_to_pandas(pandas.Series.sparse.to_coo, row_levels=row_levels, column_levels=column_levels, sort_labels=sort_labels)