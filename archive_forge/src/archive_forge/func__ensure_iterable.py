from collections import abc
import functools
import itertools
def _ensure_iterable(func):

    @functools.wraps(func)
    def wrapper(it, *args, **kwargs):
        if not isinstance(it, abc.Iterable):
            raise ValueError("Iterable expected, but '%s' is not iterable" % it)
        return func(it, *args, **kwargs)
    return wrapper