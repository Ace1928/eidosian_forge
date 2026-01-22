import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import (
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import (
from .utils import (
def _get_folded_quantized_qat_conv_bn_pattern(is_per_channel: bool, has_bias: bool, bias_is_quantized: bool, conv_fn: Callable, bn_is_training: bool) -> Callable:
    """
    Quantized QAT conv - bn pattern with bn weights being folded into conv.
    """
    bn_eps = 1e-05

    def _folded_quantized_qat_conv_bn_pattern(x: torch.Tensor, conv_weight: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor, **kwargs) -> torch.Tensor:
        conv_weight = _append_qdq(conv_weight, is_per_channel, kwargs)
        if has_bias:
            bias = kwargs['conv_bias']
            if bias_is_quantized:
                bias = _append_qdq(bias, is_per_channel, kwargs)
        else:
            bias = None
        x = conv_fn(x, conv_weight, bias)
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=bn_is_training, eps=bn_eps)
        return x
    return _folded_quantized_qat_conv_bn_pattern