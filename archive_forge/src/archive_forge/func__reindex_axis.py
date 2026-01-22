from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def _reindex_axis(obj: DataFrame, axis: AxisInt, labels: Index, other=None) -> DataFrame:
    ax = obj._get_axis(axis)
    labels = ensure_index(labels)
    if other is not None:
        other = ensure_index(other)
    if (other is None or labels.equals(other)) and labels.equals(ax):
        return obj
    labels = ensure_index(labels.unique())
    if other is not None:
        labels = ensure_index(other.unique()).intersection(labels, sort=False)
    if not labels.equals(ax):
        slicer: list[slice | Index] = [slice(None, None)] * obj.ndim
        slicer[axis] = labels
        obj = obj.loc[tuple(slicer)]
    return obj