from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def type_to_dtype(typ: type) -> torch.dtype:
    """
    Computes the corresponding dtype for a Number type.
    """
    assert isinstance(typ, type)
    if typ is bool:
        return torch.bool
    if typ in [int, torch.SymInt]:
        return torch.long
    if typ in [float, torch.SymFloat]:
        return torch.get_default_dtype()
    if typ is complex:
        return corresponding_complex_dtype(torch.get_default_dtype())
    raise ValueError('Invalid type!')