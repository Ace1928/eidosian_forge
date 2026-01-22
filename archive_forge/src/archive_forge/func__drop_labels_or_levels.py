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
def _drop_labels_or_levels(self, keys, axis: AxisInt=0):
    """
        Drop labels and/or levels for the given `axis`.

        For each key in `keys`:
          - (axis=0): If key matches a column label then drop the column.
            Otherwise if key matches an index level then drop the level.
          - (axis=1): If key matches an index label then drop the row.
            Otherwise if key matches a column level then drop the level.

        Parameters
        ----------
        keys : str or list of str
            labels or levels to drop
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        dropped: DataFrame

        Raises
        ------
        ValueError
            if any `keys` match neither a label nor a level
        """
    axis = self._get_axis_number(axis)
    keys = common.maybe_make_list(keys)
    invalid_keys = [k for k in keys if not self._is_label_or_level_reference(k, axis=axis)]
    if invalid_keys:
        raise ValueError(f'The following keys are not valid labels or levels for axis {axis}: {invalid_keys}')
    levels_to_drop = [k for k in keys if self._is_level_reference(k, axis=axis)]
    labels_to_drop = [k for k in keys if not self._is_level_reference(k, axis=axis)]
    dropped = self.copy(deep=False)
    if axis == 0:
        if levels_to_drop:
            dropped.reset_index(levels_to_drop, drop=True, inplace=True)
        if labels_to_drop:
            dropped.drop(labels_to_drop, axis=1, inplace=True)
    else:
        if levels_to_drop:
            if isinstance(dropped.columns, MultiIndex):
                dropped.columns = dropped.columns.droplevel(levels_to_drop)
            else:
                dropped.columns = RangeIndex(dropped.columns.size)
        if labels_to_drop:
            dropped.drop(labels_to_drop, axis=0, inplace=True)
    return dropped