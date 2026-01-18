import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@lazy_cache(name)
def unary_func(x):
    x = ensure_lazy(x)
    return x.to(fn=get_lib_fn(x.backend, name))