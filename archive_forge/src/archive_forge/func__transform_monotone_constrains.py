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
def _transform_monotone_constrains(self, value: Union[Dict[str, int], str, Tuple[int, ...]]) -> Union[Tuple[int, ...], str]:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return value
    constrained_features = set(value.keys())
    feature_names = self.feature_names or []
    if not constrained_features.issubset(set(feature_names)):
        raise ValueError('Constrained features are not a subset of training data feature names')
    return tuple((value.get(name, 0) for name in feature_names))