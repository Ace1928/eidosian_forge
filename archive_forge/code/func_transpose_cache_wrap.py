import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict
from .parser import alpha_canonicalize, parse_einsum_input
def transpose_cache_wrap(transpose):
    """Decorates a ``transpose()`` implementation to be memoized inside a
    :func:`shared_intermediates` context.
    """

    @functools.wraps(transpose)
    def cached_transpose(a, axes, backend='numpy'):
        if not currently_sharing():
            return transpose(a, axes, backend=backend)
        _save_tensors(a)
        axes = tuple(axes)
        key = ('transpose', backend, id(a), axes)
        return _memoize(key, transpose, a, axes, backend=backend)
    return cached_transpose