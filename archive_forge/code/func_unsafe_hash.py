import functools
from typing import Any, Callable, Dict, List, TypeVar
def unsafe_hash(obj: Any) -> Any:
    try:
        return hash(obj)
    except TypeError:
        return id(obj)