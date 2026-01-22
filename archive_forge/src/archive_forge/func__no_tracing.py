import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
def _no_tracing(func):
    """
    Decorator to temporarily turn off tracing for the duration of a test.
    Needed in tests that check refcounting, otherwise the tracing itself
    influences the refcounts
    """
    if not hasattr(sys, 'gettrace'):
        return func
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            original_trace = sys.gettrace()
            try:
                sys.settrace(None)
                return func(*args, **kwargs)
            finally:
                sys.settrace(original_trace)
        return wrapper