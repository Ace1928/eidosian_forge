import threading
from typing import Any, Dict
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(ctx, kernel_idx, grid, kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    tensors_to_clone = [key for key, value in unwrapped_kwargs.items() if isinstance(value, Tensor)]
    with ctx.redispatch_to_next():
        unwrapped_outputs = triton_kernel_wrapper_functional(kernel_idx=kernel_idx, grid=grid, kwargs=unwrapped_kwargs, tensors_to_clone=tensors_to_clone)
    assert unwrapped_outputs.keys() == kwargs.keys()
    for key, output_arg in unwrapped_outputs.items():
        if not isinstance(output_arg, Tensor):
            continue
        input_arg = kwargs[key]
        assert isinstance(input_arg, Tensor)
        ctx.replace(input_arg, output_arg)
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)
        ctx.mark_mutation_hidden_from_autograd(input_arg)
    return None