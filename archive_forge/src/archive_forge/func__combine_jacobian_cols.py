import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _combine_jacobian_cols(jacobians_cols: Dict[int, List[torch.Tensor]], outputs, input, numel) -> Tuple[torch.Tensor, ...]:
    jacobians = _allocate_jacobians_with_outputs(outputs, numel, dtype=input.dtype if input.dtype.is_complex else None)
    for i, jacobian in enumerate(jacobians):
        for k, v in jacobians_cols.items():
            jacobian[k] = v[i]
    return jacobians