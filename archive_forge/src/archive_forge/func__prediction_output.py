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
def _prediction_output(shape: CNumericPtr, dims: c_bst_ulong, predts: CFloatPtr, is_cuda: bool) -> NumpyOrCupy:
    arr_shape = ctypes2numpy(shape, dims.value, np.uint64)
    length = int(np.prod(arr_shape))
    if is_cuda:
        arr_predict = ctypes2cupy(predts, length, np.float32)
    else:
        arr_predict = ctypes2numpy(predts, length, np.float32)
    arr_predict = arr_predict.reshape(arr_shape)
    return arr_predict