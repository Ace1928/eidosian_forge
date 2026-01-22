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
def _check_label_or_level_ambiguity(self, key: Level, axis: Axis=0) -> None:
    """
        Check whether `key` is ambiguous.

        By ambiguous, we mean that it matches both a level of the input
        `axis` and a label of the other axis.

        Parameters
        ----------
        key : Hashable
            Label or level name.
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns).

        Raises
        ------
        ValueError: `key` is ambiguous
        """
    axis_int = self._get_axis_number(axis)
    other_axes = (ax for ax in range(self._AXIS_LEN) if ax != axis_int)
    if key is not None and is_hashable(key) and (key in self.axes[axis_int].names) and any((key in self.axes[ax] for ax in other_axes)):
        level_article, level_type = ('an', 'index') if axis_int == 0 else ('a', 'column')
        label_article, label_type = ('a', 'column') if axis_int == 0 else ('an', 'index')
        msg = f"'{key}' is both {level_article} {level_type} level and {label_article} {label_type} label, which is ambiguous."
        raise ValueError(msg)