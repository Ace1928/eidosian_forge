import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def to_backend_dtype(dtype_name, like):
    """Turn string specifier ``dtype_name`` into dtype of backend ``like``."""
    if not isinstance(like, str):
        like = infer_backend(like)
    try:
        return get_lib_fn(like, dtype_name)
    except ImportError:
        return dtype_name