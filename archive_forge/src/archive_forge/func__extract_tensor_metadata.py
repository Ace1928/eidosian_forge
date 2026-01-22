import torch
import torch.fx
import traceback
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional, Dict
from torch.fx._compatibility import compatibility
from torch._guards import detect_fake_mode
def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()
    memory_formats = {torch.contiguous_format, torch.channels_last, torch.channels_last_3d}
    memory_format = None
    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break
    is_quantized = result.is_quantized
    qparams: Dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams['qscheme'] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams['scale'] = result.q_scale()
            qparams['zero_point'] = result.q_zero_point()
        elif qscheme in {torch.per_channel_affine, torch.per_channel_affine_float_qparams, torch.per_channel_symmetric}:
            qparams['scale'] = result.q_per_channel_scales().tolist()
            qparams['zero_point'] = result.q_per_channel_zero_points().tolist()
            qparams['axis'] = result.q_per_channel_axis()
    return TensorMetadata(shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams)