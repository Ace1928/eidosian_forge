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
def _make_inplace(fn):
    """
    Given a function with out variant (i.e. using `out_wrapper()), it returns its in-place variant
    See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-do-in-place-operations-work-in-pytorch
    """

    @wraps(fn)
    def _fn(a, *args, **kwargs):
        return fn(a, *args, out=a, **kwargs)
    inplace_name = f'{fn.__name__}_'
    _fn.__name__ = inplace_name
    _fn = register_decomposition(getattr(aten, inplace_name))(_fn)
    from inspect import getmodule
    _all = getmodule(fn).__all__
    if inplace_name not in _all:
        _all.append(inplace_name)
    return _fn