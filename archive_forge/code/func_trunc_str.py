from inspect import (
from typing import Any, List
from .undefined import Undefined
def trunc_str(s: str) -> str:
    """Truncate strings to maximum length."""
    if len(s) > max_str_size:
        i = max(0, (max_str_size - 3) // 2)
        j = max(0, max_str_size - 3 - i)
        s = s[:i] + '...' + s[-j:]
    return s