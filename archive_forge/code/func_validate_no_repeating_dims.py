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
def validate_no_repeating_dims(dims: Sequence):
    if len(dims) != len(set(dims)):
        raise RuntimeError('duplicate value in the list of dims')