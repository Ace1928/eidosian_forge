from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Literal, TypedDict
from xarray.core.utils import FrozenDict
def _set_file_cache_maxsize(value) -> None:
    from xarray.backends.file_manager import FILE_CACHE
    FILE_CACHE.maxsize = value