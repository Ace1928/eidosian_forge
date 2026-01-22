from __future__ import annotations
import typing as T
from .baseobjects import MesonInterpreterObject
def _is_arg_disabled(arg: T.Any) -> bool:
    if isinstance(arg, Disabler):
        return True
    if isinstance(arg, list):
        for i in arg:
            if _is_arg_disabled(i):
                return True
    return False