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
def _annotate_conv2d(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
    conv_partitions = get_source_partitions(gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d])
    conv_partitions = list(itertools.chain(*conv_partitions.values()))
    for conv_partition in conv_partitions:
        if len(conv_partition.output_nodes) > 1:
            raise ValueError('conv partition has more than one output node')
        conv_node = conv_partition.output_nodes[0]
        if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
            raise ValueError(f'{conv_node} is not an aten conv2d operator')
        if _is_annotated([conv_node]):
            continue
        self._annotate_conv_node_helper(conv_node, True, quantization_config)