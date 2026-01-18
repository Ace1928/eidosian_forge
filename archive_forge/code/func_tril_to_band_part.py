import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tril_to_band_part(fn):

    @functools.wraps(fn)
    def numpy_like(x, k=0):
        if k < 0:
            raise ValueError("'k' must be positive to recreate 'numpy.tril' behaviour with 'tensorflow.matrix_band_part'.")
        return fn(x, -1, k)
    return numpy_like