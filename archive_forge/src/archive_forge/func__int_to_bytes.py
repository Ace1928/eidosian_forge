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
def _int_to_bytes(i):
    num_bytes = (i.bit_length() + 8) // 8
    return i.to_bytes(num_bytes, 'little', signed=True)