import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@shape.register('torch')
def torch_shape(x):
    return tuple(x.shape)