from __future__ import annotations
import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload
def _get_stack_depth(size: int=2) -> int:
    """Get current stack depth, performantly.
    """
    frame = sys._getframe(size)
    for size in count(size):
        frame = frame.f_back
        if not frame:
            return size