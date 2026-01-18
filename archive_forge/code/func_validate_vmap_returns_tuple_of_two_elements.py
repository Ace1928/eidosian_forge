import torch
from torch._ops import HigherOrderOperator
from torch._C._functorch import TransformType
from torch._functorch.utils import enable_single_level_autograd_function
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch._functorch.vmap import (
from torch._functorch.apis import vmap
from torch._functorch.vmap import _broadcast_to_and_flatten
from torch.autograd.forward_ad import _set_fwd_grad_enabled
from typing import Any, NamedTuple, Tuple
def validate_vmap_returns_tuple_of_two_elements(result):
    base_error_msg = 'Expected the vmap staticmethod to have two returns, an output and out_dims with pytree structure compatible with the output. '
    if not isinstance(result, tuple):
        raise RuntimeError(base_error_msg + f'Got a {type(result)} instead')
    if not len(result) == 2:
        raise RuntimeError(base_error_msg + f'Got {len(result)} returns instead')