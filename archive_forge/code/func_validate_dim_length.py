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
def validate_dim_length(length: int):
    """
    Validates that an object represents a valid
    dimension length.
    """
    if isinstance(length, (int, torch.SymInt)):
        torch._check_is_size(length)
    else:
        assert length >= 0