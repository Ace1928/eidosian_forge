import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_astype(x, dtype):
    return x.to(dtype=to_backend_dtype(dtype, like=x))