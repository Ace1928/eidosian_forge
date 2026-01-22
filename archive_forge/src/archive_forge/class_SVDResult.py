from __future__ import annotations
import functools
from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._manipulation_functions import reshape
from ._array_object import Array
from .._core.internal import _normalize_axis_indices as normalize_axis_tuple
from typing import TYPE_CHECKING
from typing import NamedTuple
import cupy as np
class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array