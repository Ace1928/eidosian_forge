import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_tensordot_wrap(fn):

    @functools.wraps(fn)
    def numpy_like(a, b, axes=2):
        return fn(a, b, dims=axes)
    return numpy_like