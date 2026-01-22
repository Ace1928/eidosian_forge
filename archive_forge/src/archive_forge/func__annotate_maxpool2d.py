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
def _annotate_maxpool2d(self, node: Node, quantization_config: QuantizationConfig) -> None:
    if node.target is not torch.ops.aten.max_pool2d.default:
        return
    maxpool_node = node
    if _is_any_annotated([maxpool_node]):
        return
    input_node = maxpool_node.args[0]
    assert isinstance(input_node, Node)
    input_qspec_map = {}
    input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
    maxpool_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)