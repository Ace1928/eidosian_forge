import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def _add_sharing_cache(cache):
    _SHARING_STACK[threading.get_ident()].append(cache)