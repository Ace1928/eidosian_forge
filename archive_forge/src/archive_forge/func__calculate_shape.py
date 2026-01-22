import warnings
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, Union
import torch
from torch.types import _size, _TensorOrTensors, _TensorOrTensorsOrGradEdge
from .. import _vmap_internals
from ..overrides import handle_torch_function, has_torch_function, is_tensor_like
from . import forward_ad, functional, graph
from .anomaly_mode import detect_anomaly, set_detect_anomaly
from .function import Function, NestedIOFunction
from .grad_mode import (
from .gradcheck import gradcheck, gradgradcheck
from .variable import Variable
from torch._C._autograd import (
from torch._C._profiler import ProfilerActivity, ProfilerConfig, ProfilerState
from . import profiler
def _calculate_shape(output: torch.Tensor, grad: torch.Tensor, is_grads_batched: bool) -> Tuple[_ShapeorNestedShape, _ShapeorNestedShape]:
    from torch.nested._internal.nested_tensor import NestedTensor
    if output.is_nested and (not isinstance(output, NestedTensor)):
        if is_grads_batched:
            raise RuntimeError('Batched grads are not supported with Nested Tensor.')
        out_shape = output._nested_tensor_size()
        grad_shape = grad._nested_tensor_size()
        return (out_shape, grad_shape)
    reg_out_shape = output.shape
    reg_grad_shape = grad.shape if not is_grads_batched else grad.shape[1:]
    return (reg_out_shape, reg_grad_shape)