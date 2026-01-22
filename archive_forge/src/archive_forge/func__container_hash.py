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
def _container_hash(obj):
    h = hashlib.new('md5')
    h.update(_generate_hash(f'__{type(obj).__name__}'))
    for item in obj.items() if isinstance(obj, dict) else obj:
        h.update(_generate_hash(item))
    return h.digest()