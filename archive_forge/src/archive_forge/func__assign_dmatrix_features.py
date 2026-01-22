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
def _assign_dmatrix_features(self, data: DMatrix) -> None:
    if data.num_row() == 0:
        return
    fn = data.feature_names
    ft = data.feature_types
    if self.feature_names is None:
        self.feature_names = fn
    if self.feature_types is None:
        self.feature_types = ft
    self._validate_features(fn)