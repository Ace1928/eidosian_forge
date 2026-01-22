from __future__ import annotations
import logging # isort:skip
import datetime as dt
import uuid
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any
import numpy as np
from ..core.types import ID
from ..settings import settings
from .strings import format_docstring
def _cast_if_can(array: npt.NDArray[Any], dtype: type[Any]) -> npt.NDArray[Any]:
    info = np.iinfo(dtype)
    if np.any((array < info.min) | (info.max < array)):
        return array
    else:
        return array.astype(dtype, casting='unsafe')