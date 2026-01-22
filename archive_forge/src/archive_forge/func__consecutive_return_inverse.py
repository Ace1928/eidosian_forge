from typing import (
import operator
import itertools
import torch
from torch._C import _add_docstr
import torch.nn.functional as F
from ._lowrank import svd_lowrank, pca_lowrank
from .overrides import (
from ._jit_internal import boolean_dispatch
from ._jit_internal import _overload as overload
from torch import _VF
def _consecutive_return_inverse(input, return_inverse=False, return_counts=False, dim=None):
    if has_torch_function_unary(input):
        return _unique_consecutive_impl(input, return_inverse, return_counts, dim)
    output, inverse_indices, _ = _unique_consecutive_impl(input, return_inverse, return_counts, dim)
    return (output, inverse_indices)