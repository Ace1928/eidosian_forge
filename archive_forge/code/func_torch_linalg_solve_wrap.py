import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_linalg_solve_wrap(fn):

    @binary_allow_1d_rhs_wrap
    def numpy_like(a, b):
        return fn(b, a)[0]
    return numpy_like