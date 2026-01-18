import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_count_nonzero(x):
    return do('sum', x != 0, like='torch')