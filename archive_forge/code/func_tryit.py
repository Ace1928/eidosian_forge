from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def tryit(rule: Callable[[_T], _T], exception) -> Callable[[_T], _T]:
    """ Return original expr if rule raises exception """

    def try_rl(expr: _T) -> _T:
        try:
            return rule(expr)
        except exception:
            return expr
    return try_rl