import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def _get_log_callback_func() -> Callable:
    """Wrap log_callback() method in ctypes callback type"""
    c_callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
    return c_callback(_log_callback)