from __future__ import annotations
from operator import (
import textwrap
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.tslibs import (
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
from pandas.util._exceptions import rewrite_exception
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.extension import (
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
def _get_indexer_pointwise(self, target: Index) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
        pointwise implementation for get_indexer and get_indexer_non_unique.
        """
    indexer, missing = ([], [])
    for i, key in enumerate(target):
        try:
            locs = self.get_loc(key)
            if isinstance(locs, slice):
                locs = np.arange(locs.start, locs.stop, locs.step, dtype='intp')
            elif lib.is_integer(locs):
                locs = np.array(locs, ndmin=1)
            else:
                locs = np.where(locs)[0]
        except KeyError:
            missing.append(i)
            locs = np.array([-1])
        except InvalidIndexError:
            missing.append(i)
            locs = np.array([-1])
        indexer.append(locs)
    indexer = np.concatenate(indexer)
    return (ensure_platform_int(indexer), ensure_platform_int(missing))