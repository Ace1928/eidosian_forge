from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def try_rl(expr: _T) -> _T:
    try:
        return rule(expr)
    except exception:
        return expr