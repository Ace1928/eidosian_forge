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
def validate_shape(shape: ShapeType):
    """
    Validates that a sequence represents a valid shape.
    """
    assert isinstance(shape, Sequence), type(shape)
    for l in shape:
        validate_dim_length(l)