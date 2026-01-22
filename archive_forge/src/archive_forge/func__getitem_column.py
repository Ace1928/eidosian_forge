from __future__ import annotations
import datetime
import functools
import itertools
import os
import re
import sys
import warnings
from typing import (
import numpy as np
import pandas
from pandas._libs import lib
from pandas._typing import (
from pandas.core.common import apply_if_callable, get_cython_func
from pandas.core.computation.eval import _check_engine
from pandas.core.dtypes.common import (
from pandas.core.indexes.frozen import FrozenList
from pandas.io.formats.info import DataFrameInfo
from pandas.util._validators import validate_bool_kwarg
from modin.config import PersistentPickle
from modin.error_message import ErrorMessage
from modin.logging import disable_logging
from modin.pandas import Categorical
from modin.pandas.io import from_non_pandas, from_pandas, to_pandas
from modin.utils import (
from .accessor import CachedAccessor, SparseFrameAccessor
from .base import _ATTRS_NO_LOOKUP, BasePandasDataset
from .groupby import DataFrameGroupBy
from .iterator import PartitionIterator
from .series import Series
from .utils import (
def _getitem_column(self, key) -> Series:
    """
        Get column specified by `key`.

        Parameters
        ----------
        key : hashable
            Key that points to column to retrieve.

        Returns
        -------
        Series
            Selected column.
        """
    if key not in self.keys():
        raise KeyError('{}'.format(key))
    s = self.__constructor__(query_compiler=self._query_compiler.getitem_column_array([key])).squeeze(axis=1)
    if isinstance(s, Series):
        s._parent = self
        s._parent_axis = 1
    return s