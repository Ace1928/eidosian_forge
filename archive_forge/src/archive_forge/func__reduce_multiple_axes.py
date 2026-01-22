from __future__ import annotations
from functools import wraps
from builtins import all as builtin_all, any as builtin_any
from ..common._aliases import (UniqueAllResult, UniqueCountsResult,
from .._internal import get_xp
import torch
from typing import TYPE_CHECKING
def _reduce_multiple_axes(f, x, axis, keepdims=False, **kwargs):
    axes = _normalize_axes(axis, x.ndim)
    for a in reversed(axes):
        x = torch.movedim(x, a, -1)
    x = torch.flatten(x, -len(axes))
    out = f(x, -1, **kwargs)
    if keepdims:
        for a in axes:
            out = torch.unsqueeze(out, a)
    return out