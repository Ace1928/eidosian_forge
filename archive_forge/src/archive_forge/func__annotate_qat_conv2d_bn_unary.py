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
def _annotate_qat_conv2d_bn_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
    fused_partitions = []
    unary_patterns = [[torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU], [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Hardtanh], [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU6]]
    for unary_pattern in unary_patterns:
        partitions = find_sequential_partitions(gm, unary_pattern)
        if partitions:
            fused_partitions.extend(partitions)
    for fused_partition in fused_partitions:
        conv_partition, bn_partition, unary_partition = fused_partition
        conv_node, bn_output_node, unary_node = self._get_output_nodes_of_partitions([conv_partition, bn_partition, unary_partition])
        if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
            continue
        if _is_annotated([unary_node, bn_output_node, conv_node]):
            continue
        self._annotate_conv_node_helper(conv_node, False, quantization_config)
        unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True, _is_output_of_quantized_pattern=True)
        nodes_to_mark_annotated = list(conv_partition.nodes)
        nodes_to_mark_annotated.extend(list(bn_partition.nodes))
        nodes_to_mark_annotated.extend(list(unary_partition.nodes))
        _mark_nodes_as_annotated(nodes_to_mark_annotated)