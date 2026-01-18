import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_zeros_ones_wrap(fn):

    @functools.wraps(fn)
    def numpy_like(shape, dtype=None, **kwargs):
        if dtype is not None:
            dtype = to_backend_dtype(dtype, like='torch')
        return fn(shape, dtype=dtype)
    return numpy_like