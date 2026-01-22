from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
def _align_series(self, other: Series, join: AlignJoin='outer', axis: Axis | None=None, level=None, copy: bool_t | None=None, fill_value=None, method=None, limit: int | None=None, fill_axis: Axis=0) -> tuple[Self, Series, Index | None]:
    is_series = isinstance(self, ABCSeries)
    if copy and using_copy_on_write():
        copy = False
    if not is_series and axis is None or axis not in [None, 0, 1]:
        raise ValueError('Must specify axis=0 or 1')
    if is_series and axis == 1:
        raise ValueError('cannot align series to a series other than axis 0')
    if not axis:
        if self.index.equals(other.index):
            join_index, lidx, ridx = (None, None, None)
        else:
            join_index, lidx, ridx = self.index.join(other.index, how=join, level=level, return_indexers=True)
        if is_series:
            left = self._reindex_indexer(join_index, lidx, copy)
        elif lidx is None or join_index is None:
            left = self.copy(deep=copy)
        else:
            new_mgr = self._mgr.reindex_indexer(join_index, lidx, axis=1, copy=copy)
            left = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        right = other._reindex_indexer(join_index, ridx, copy)
    else:
        fdata = self._mgr
        join_index = self.axes[1]
        lidx, ridx = (None, None)
        if not join_index.equals(other.index):
            join_index, lidx, ridx = join_index.join(other.index, how=join, level=level, return_indexers=True)
        if lidx is not None:
            bm_axis = self._get_block_manager_axis(1)
            fdata = fdata.reindex_indexer(join_index, lidx, axis=bm_axis)
        if copy and fdata is self._mgr:
            fdata = fdata.copy()
        left = self._constructor_from_mgr(fdata, axes=fdata.axes)
        if ridx is None:
            right = other.copy(deep=copy)
        else:
            right = other.reindex(join_index, level=level)
    fill_na = notna(fill_value) or method is not None
    if fill_na:
        fill_value, method = validate_fillna_kwargs(fill_value, method)
        if method is not None:
            left = left._pad_or_backfill(method, limit=limit, axis=fill_axis)
            right = right._pad_or_backfill(method, limit=limit)
        else:
            left = left.fillna(fill_value, limit=limit, axis=fill_axis)
            right = right.fillna(fill_value, limit=limit)
    return (left, right, join_index)