from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Literal, TypedDict
from xarray.core.utils import FrozenDict
def _get_keep_attrs(default: bool) -> bool:
    return _get_boolean_with_default('keep_attrs', default)