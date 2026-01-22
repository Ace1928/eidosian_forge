from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _interval_to_bound_points(array: Sequence[pd.Interval]) -> np.ndarray:
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """
    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate((array_boundaries, np.array([array[-1].right])))
    return array_boundaries