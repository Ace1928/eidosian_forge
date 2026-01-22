import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@functools.lru_cache(None)
def _find_common_backend_cached(names):
    return max(names, key=lambda n: multi_class_priorities.get(n, 0))