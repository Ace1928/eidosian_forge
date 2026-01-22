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
def _lib_version(lib: ctypes.CDLL) -> Tuple[int, int, int]:
    """Get the XGBoost version from native shared object."""
    major = ctypes.c_int()
    minor = ctypes.c_int()
    patch = ctypes.c_int()
    lib.XGBoostVersion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
    return (major.value, minor.value, patch.value)