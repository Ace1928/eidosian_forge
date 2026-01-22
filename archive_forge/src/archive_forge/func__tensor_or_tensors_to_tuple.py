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
def _tensor_or_tensors_to_tuple(tensors: Optional[_TensorOrTensors], length: int) -> Tuple[_OptionalTensor, ...]:
    if tensors is None:
        return (None,) * length
    if isinstance(tensors, torch.Tensor):
        return (tensors,)
    return tuple(tensors)