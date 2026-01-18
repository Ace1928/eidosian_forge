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
def validate_memory_format(memory_format: torch.memory_format):
    torch._check(memory_format in _memory_formats, lambda: f'Received unknown memory format {memory_format}!')