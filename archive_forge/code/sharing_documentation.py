import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict
from .parser import alpha_canonicalize, parse_einsum_input
Decorates an ``to_backend()`` implementation to be memoized inside a
    :func:`shared_intermediates` context (e.g. ``to_cupy``, ``to_torch``).
    