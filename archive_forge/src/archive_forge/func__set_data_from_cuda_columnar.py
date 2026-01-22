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
def _set_data_from_cuda_columnar(self, data: DataType, cat_codes: list) -> None:
    """Set data from CUDA columnar format."""
    from .data import _cudf_array_interfaces
    interfaces_str = _cudf_array_interfaces(data, cat_codes)
    _check_call(_LIB.XGProxyDMatrixSetDataCudaColumnar(self.handle, interfaces_str))