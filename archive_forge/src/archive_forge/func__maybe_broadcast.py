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
def _maybe_broadcast(*args, preserve_cpu_scalar_tensors=True):
    common_shape = _broadcast_shapes(*(t.shape if isinstance(t, TensorLike) else None for t in args))

    def __maybe_broadcast(x, shape):
        if x is None:
            return None
        elif isinstance(x, Number):
            return x
        elif isinstance(x, TensorLike):
            if preserve_cpu_scalar_tensors and utils.is_cpu_scalar_tensor(x):
                return x
            if not utils.same_shape(x.shape, common_shape):
                return x.expand(common_shape)
            return x
        else:
            raise RuntimeError('Unexpected type when broadcasting: ' + str(type(x)) + '!')
    return tuple((__maybe_broadcast(x, common_shape) for x in args))