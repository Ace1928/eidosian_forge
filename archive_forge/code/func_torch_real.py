import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_real(x):
    try:
        if x.is_complex():
            return x.real
    except AttributeError:
        pass
    return x