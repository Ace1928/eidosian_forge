from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Literal, TypedDict
from xarray.core.utils import FrozenDict
def _get_boolean_with_default(option: Options, default: bool) -> bool:
    global_choice = OPTIONS[option]
    if global_choice == 'default':
        return default
    elif isinstance(global_choice, bool):
        return global_choice
    else:
        raise ValueError(f"The global option {option} must be one of True, False or 'default'.")