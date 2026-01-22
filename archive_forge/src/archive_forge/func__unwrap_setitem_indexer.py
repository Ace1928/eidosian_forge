from __future__ import annotations
from functools import wraps
import inspect
import re
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv
def _unwrap_setitem_indexer(self, indexer):
    """
        Adapt a 2D-indexer to our 1D values.

        This is intended for 'setitem', not 'iget' or '_slice'.
        """
    if isinstance(indexer, tuple) and len(indexer) == 2:
        if all((isinstance(x, np.ndarray) and x.ndim == 2 for x in indexer)):
            first, second = indexer
            if not (second.size == 1 and (second == 0).all() and (first.shape[1] == 1)):
                raise NotImplementedError('This should not be reached. Please report a bug at github.com/pandas-dev/pandas/')
            indexer = first[:, 0]
        elif lib.is_integer(indexer[1]) and indexer[1] == 0:
            indexer = indexer[0]
        elif com.is_null_slice(indexer[1]):
            indexer = indexer[0]
        elif is_list_like(indexer[1]) and indexer[1][0] == 0:
            indexer = indexer[0]
        else:
            raise NotImplementedError('This should not be reached. Please report a bug at github.com/pandas-dev/pandas/')
    return indexer