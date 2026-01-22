import functools
from typing import Callable, TypeVar
from cvxpy.utilities import scopes
@functools.wraps(func)
def _compute_once(self, *args, **kwargs) -> R:
    cache_name = func.__name__ + '__cache__'
    if not hasattr(self, cache_name):
        setattr(self, cache_name, {})
    cache = getattr(self, cache_name)
    key = _cache_key(args, kwargs)
    if key in cache:
        return cache[key]
    result = func(self, *args, **kwargs)
    cache[key] = result
    return result