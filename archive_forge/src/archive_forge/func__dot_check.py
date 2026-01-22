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
def _dot_check(self, other):
    torch._check(self.dim() == 1 and other.dim() == 1, lambda: f'1D tensors expected, but got {self.dim()}D and {other.dim()}D tensors')

    def numel_error():
        return f'inconsistent tensor size, expected tensor [{self.numel()}] and src [{other.numel()}] to have thesame number of elements, but got {self.numel()} and {other.numel()} elements respectively'
    torch._check(self.numel() == other.numel(), numel_error)