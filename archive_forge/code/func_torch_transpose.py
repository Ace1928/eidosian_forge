import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_transpose(x, axes=None):
    if axes is None:
        axes = reversed(range(0, x.ndimension()))
    return x.permute(*axes)