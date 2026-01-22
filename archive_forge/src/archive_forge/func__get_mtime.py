import collections
import contextlib
import cProfile
import inspect
import gc
import multiprocessing
import os
import random
import sys
import time
import unittest
import warnings
import zlib
from functools import lru_cache
from io import StringIO
from unittest import result, runner, signals, suite, loader, case
from .loader import TestLoader
from numba.core import config
@lru_cache(maxsize=128)
def _get_mtime(cls):
    """
    Gets the mtime of the file in which a test class is defined.
    """
    return str(os.path.getmtime(inspect.getfile(cls)))