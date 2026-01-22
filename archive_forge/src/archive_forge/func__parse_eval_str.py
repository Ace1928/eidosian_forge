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
def _parse_eval_str(result: str) -> List[Tuple[str, float]]:
    """Parse an eval result string from the booster."""
    splited = result.split()[1:]
    metric_score_str = [tuple(s.split(':')) for s in splited]
    metric_score = [(n, float(s)) for n, s in metric_score_str]
    return metric_score