from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import (
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
def _decompose_slice(key: slice, size: int) -> tuple[slice, slice]:
    """convert a slice to successive two slices. The first slice always has
    a positive step.

    >>> _decompose_slice(slice(2, 98, 2), 99)
    (slice(2, 98, 2), slice(None, None, None))

    >>> _decompose_slice(slice(98, 2, -2), 99)
    (slice(4, 99, 2), slice(None, None, -1))

    >>> _decompose_slice(slice(98, 2, -2), 98)
    (slice(3, 98, 2), slice(None, None, -1))

    >>> _decompose_slice(slice(360, None, -10), 361)
    (slice(0, 361, 10), slice(None, None, -1))
    """
    start, stop, step = key.indices(size)
    if step > 0:
        return (key, slice(None))
    else:
        exact_stop = range(start, stop, step)[-1]
        return (slice(exact_stop, start + 1, -step), slice(None, None, -1))