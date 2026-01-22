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
class IndexCallable:
    """Provide getitem and setitem syntax for callable objects."""
    __slots__ = ('getter', 'setter')

    def __init__(self, getter: Callable[..., Any], setter: Callable[..., Any] | None=None):
        self.getter = getter
        self.setter = setter

    def __getitem__(self, key: Any) -> Any:
        return self.getter(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if self.setter is None:
            raise NotImplementedError('Setting values is not supported for this indexer.')
        self.setter(key, value)