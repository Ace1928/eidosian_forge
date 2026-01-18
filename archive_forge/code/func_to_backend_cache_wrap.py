import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict
from .parser import alpha_canonicalize, parse_einsum_input
def to_backend_cache_wrap(to_backend=None, constants=False):
    """Decorates an ``to_backend()`` implementation to be memoized inside a
    :func:`shared_intermediates` context (e.g. ``to_cupy``, ``to_torch``).
    """
    if to_backend is None:
        return functools.partial(to_backend_cache_wrap, constants=constants)
    if constants:

        @functools.wraps(to_backend)
        def cached_to_backend(array, constant=False):
            if not currently_sharing():
                return to_backend(array, constant=constant)
            key = (to_backend.__name__, id(array), constant)
            return _memoize(key, to_backend, array, constant=constant)
    else:

        @functools.wraps(to_backend)
        def cached_to_backend(array):
            if not currently_sharing():
                return to_backend(array)
            key = (to_backend.__name__, id(array))
            return _memoize(key, to_backend, array)
    return cached_to_backend