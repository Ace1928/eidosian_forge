from __future__ import annotations
import contextlib
import functools
import math
from collections import defaultdict
from collections.abc import Collection, Hashable, Sequence
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from reprlib import recursive_repr
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from xarray.core.duck_array_ops import array_equiv, astype
from xarray.core.indexing import MemoryCachedArray
from xarray.core.options import OPTIONS, _get_boolean_with_default
from xarray.core.utils import is_duck_array
from xarray.namedarray.pycompat import array_type, to_duck_array, to_numpy
def unindexed_dims_repr(dims, coords, max_rows: int | None=None):
    unindexed_dims = [d for d in dims if d not in coords]
    if unindexed_dims:
        dims_start = 'Dimensions without coordinates: '
        dims_str = _element_formatter(unindexed_dims, col_width=len(dims_start), max_rows=max_rows)
        return dims_start + dims_str
    else:
        return None