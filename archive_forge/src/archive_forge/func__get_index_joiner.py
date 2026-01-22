from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def _get_index_joiner(self, index_cls) -> Callable:
    if self.join in ['outer', 'inner']:
        return functools.partial(functools.reduce, functools.partial(index_cls.join, how=self.join))
    elif self.join == 'left':
        return operator.itemgetter(0)
    elif self.join == 'right':
        return operator.itemgetter(-1)
    elif self.join == 'override':
        return operator.itemgetter(0)
    else:
        return lambda _: None