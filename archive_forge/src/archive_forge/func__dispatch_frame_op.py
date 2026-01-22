from __future__ import annotations
import collections
from collections import abc
from collections.abc import (
import functools
from inspect import signature
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.util._validators import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import (
from pandas.core.indexes.multi import (
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.io.common import get_handle
from pandas.io.formats import (
from pandas.io.formats.info import (
import pandas.plotting
def _dispatch_frame_op(self, right, func: Callable, axis: AxisInt | None=None) -> DataFrame:
    """
        Evaluate the frame operation func(left, right) by evaluating
        column-by-column, dispatching to the Series implementation.

        Parameters
        ----------
        right : scalar, Series, or DataFrame
        func : arithmetic or comparison operator
        axis : {None, 0, 1}

        Returns
        -------
        DataFrame

        Notes
        -----
        Caller is responsible for setting np.errstate where relevant.
        """
    array_op = ops.get_array_op(func)
    right = lib.item_from_zerodim(right)
    if not is_list_like(right):
        bm = self._mgr.apply(array_op, right=right)
        return self._constructor_from_mgr(bm, axes=bm.axes)
    elif isinstance(right, DataFrame):
        assert self.index.equals(right.index)
        assert self.columns.equals(right.columns)
        bm = self._mgr.operate_blockwise(right._mgr, array_op)
        return self._constructor_from_mgr(bm, axes=bm.axes)
    elif isinstance(right, Series) and axis == 1:
        assert right.index.equals(self.columns)
        right = right._values
        assert not isinstance(right, np.ndarray)
        arrays = [array_op(_left, _right) for _left, _right in zip(self._iter_column_arrays(), right)]
    elif isinstance(right, Series):
        assert right.index.equals(self.index)
        right = right._values
        arrays = [array_op(left, right) for left in self._iter_column_arrays()]
    else:
        raise NotImplementedError(right)
    return type(self)._from_arrays(arrays, self.columns, self.index, verify_integrity=False)