from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from .dataframe import DataFrame
from .series import Series
from .utils import is_scalar
def _validate_key_length(self, key: tuple) -> tuple:
    if len(key) > self.df.ndim:
        if key[0] is Ellipsis:
            key = key[1:]
            if Ellipsis in key:
                raise IndexingError(_one_ellipsis_message)
            return self._validate_key_length(key)
        raise IndexingError('Too many indexers')
    return key