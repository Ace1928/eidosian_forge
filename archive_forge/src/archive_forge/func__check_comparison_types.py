from __future__ import annotations
import operator
import re
from re import Pattern
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import isna
def _check_comparison_types(result: ArrayLike | bool, a: ArrayLike, b: Scalar | Pattern):
    """
        Raises an error if the two arrays (a,b) cannot be compared.
        Otherwise, returns the comparison result as expected.
        """
    if is_bool(result) and isinstance(a, np.ndarray):
        type_names = [type(a).__name__, type(b).__name__]
        type_names[0] = f'ndarray(dtype={a.dtype})'
        raise TypeError(f'Cannot compare types {repr(type_names[0])} and {repr(type_names[1])}')