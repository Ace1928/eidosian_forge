from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def _has_infs(result) -> bool:
    if isinstance(result, np.ndarray):
        if result.dtype in ('f8', 'f4'):
            return lib.has_infs(result.ravel('K'))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        return False