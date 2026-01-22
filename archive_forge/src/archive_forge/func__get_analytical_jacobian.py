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
def _get_analytical_jacobian(inputs, outputs, input_idx, output_idx):
    jacobians = _check_analytical_jacobian_attributes(inputs, outputs[output_idx], nondet_tol=float('inf'), check_grad_dtypes=False)
    return jacobians[input_idx]