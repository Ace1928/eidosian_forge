from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def validate_limit_area(limit_area: str | None) -> Literal['inside', 'outside'] | None:
    if limit_area is not None:
        valid_limit_areas = ['inside', 'outside']
        limit_area = limit_area.lower()
        if limit_area not in valid_limit_areas:
            raise ValueError(f'Invalid limit_area: expecting one of {valid_limit_areas}, got {limit_area}.')
    return limit_area