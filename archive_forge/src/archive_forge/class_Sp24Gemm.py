import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
@register_operator
class Sp24Gemm(BaseOperator):
    OPERATOR = get_xformers_operator('_sparse24_gemm')
    OPERATOR_CATEGORY = 'sp24'
    NAME = '_sparse24_gemm'