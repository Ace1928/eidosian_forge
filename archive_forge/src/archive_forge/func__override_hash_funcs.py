import datetime as dt
import functools
import hashlib
import inspect
import io
import os
import pathlib
import pickle
import sys
import threading
import time
import unittest
import unittest.mock
import weakref
from contextlib import contextmanager
import param
from param.parameterized import iscoroutinefunction
from .state import state
@contextmanager
def _override_hash_funcs(hash_funcs):
    backup = dict(_hash_funcs)
    _hash_funcs.update(hash_funcs)
    try:
        yield
    finally:
        _hash_funcs.clear()
        _hash_funcs.update(backup)