import threading
from typing import Any, Dict
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
@triton_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(mode, *, kernel_idx, grid, kwargs, tensors_to_clone):
    if mode.enable_tracing:
        return trace_triton_kernel_wrapper(mode, triton_kernel_wrapper_functional, {'kernel_idx': kernel_idx, 'grid': grid, 'kwargs': kwargs, 'tensors_to_clone': tensors_to_clone})
    else:
        return triton_kernel_wrapper_functional(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs, tensors_to_clone=tensors_to_clone)