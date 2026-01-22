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
def _allocate_jacobians_with_outputs(output_tensors: Tuple, numel_input, dtype=None, device=None) -> Tuple[torch.Tensor, ...]:
    out: List[torch.Tensor] = []
    options = {'dtype': dtype, 'device': device, 'layout': torch.strided}
    for t in output_tensors:
        if _is_float_or_complex_tensor(t):
            out.append(t.new_zeros((numel_input, t.numel()), **options))
    return tuple(out)