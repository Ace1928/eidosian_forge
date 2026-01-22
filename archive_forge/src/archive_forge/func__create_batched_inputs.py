import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def _create_batched_inputs(flat_in_dims: List[Any], flat_args: List[Any], vmap_level: int, args_spec) -> Tuple:
    batched_inputs = [arg if in_dim is None else _add_batch_dim(arg, in_dim, vmap_level) for in_dim, arg in zip(flat_in_dims, flat_args)]
    return tree_unflatten(batched_inputs, args_spec)