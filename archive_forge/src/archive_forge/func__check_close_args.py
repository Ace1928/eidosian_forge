import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def _check_close_args(name: str, a: TensorLikeType, b: TensorLikeType, rtol: float, atol: float) -> None:
    torch._check_value(a.dtype == b.dtype, lambda: f'{name}: Attempting to compare tensors of different dtypes {a.dtype} and {b.dtype}!')
    torch._check(rtol >= 0, lambda: f'{name}: rtol must be greater than or equal to zero, but got {rtol}!')
    torch._check(atol >= 0, lambda: f'{name}: atol must be greater than or equal to zero, but got {atol}!')