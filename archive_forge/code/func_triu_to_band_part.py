import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def triu_to_band_part(fn):

    @functools.wraps(fn)
    def numpy_like(x, k=0):
        if k > 0:
            raise ValueError("'k' must be negative to recreate 'numpy.triu' behaviour with 'tensorflow.matrix_band_part'.")
        return fn(x, -k, -1)
    return numpy_like