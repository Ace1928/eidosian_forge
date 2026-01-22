import copy
import functools
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
def _annotate_linear_node_helper(self, linear_node: torch.fx.Node, annotate_output: bool, quantization_config: QuantizationConfig) -> None:
    """Helper function to annotate the linear node"""
    input_qspec_map = {}
    assert linear_node.target in (torch.ops.aten.linear.default,)
    has_bias = len(linear_node.args) == 3
    input_index = 0
    weight_index = 1
    bias_index = 2
    input_node = linear_node.args[input_index]
    assert isinstance(input_node, Node)
    input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
    weight_node = linear_node.args[weight_index]
    assert isinstance(weight_node, Node)
    input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
    bias_node = linear_node.args[bias_index] if has_bias else None
    if isinstance(bias_node, Node):
        input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
    if annotate_output:
        linear_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)
    else:
        linear_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)