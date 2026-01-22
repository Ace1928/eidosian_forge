from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def _conv_corr_impl(a, v, mode):
    dt = _dtypes_impl.result_type_impl(a, v)
    a = _util.cast_if_needed(a, dt)
    v = _util.cast_if_needed(v, dt)
    padding = v.shape[0] - 1 if mode == 'full' else mode
    if padding == 'same' and v.shape[0] % 2 == 0:
        raise NotImplementedError("mode='same' and even-length weights")
    aa = a[None, :]
    vv = v[None, None, :]
    result = torch.nn.functional.conv1d(aa, vv, padding=padding)
    return result[0, :]