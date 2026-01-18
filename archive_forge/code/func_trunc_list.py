from inspect import (
from typing import Any, List
from .undefined import Undefined
def trunc_list(s: List) -> List:
    """Truncate lists to maximum length."""
    if len(s) > max_list_size:
        i = max_list_size // 2
        j = i - 1
        s = s[:i] + [ELLIPSIS] + s[-j:]
    return s