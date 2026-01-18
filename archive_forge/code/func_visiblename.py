import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def visiblename(name, all=None, obj=None):
    """Decide whether to show documentation on a variable."""
    if name in {'__author__', '__builtins__', '__cached__', '__credits__', '__date__', '__doc__', '__file__', '__spec__', '__loader__', '__module__', '__name__', '__package__', '__path__', '__qualname__', '__slots__', '__version__'}:
        return 0
    if name.startswith('__') and name.endswith('__'):
        return 1
    if name.startswith('_') and hasattr(obj, '_fields'):
        return True
    if obj is not __future__ and name in _future_feature_names:
        if isinstance(getattr(obj, name, None), __future__._Feature):
            return False
    if all is not None:
        return name in all
    else:
        return not name.startswith('_')