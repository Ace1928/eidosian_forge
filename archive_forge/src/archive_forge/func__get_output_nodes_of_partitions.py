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
def _get_output_nodes_of_partitions(self, partition_list: List[SourcePartition]) -> List[torch.fx.Node]:
    """Helper function to get the output node list from partition list"""
    output_node_list = []
    for partition in partition_list:
        if len(partition.output_nodes) > 1:
            raise ValueError('Input partition has more than one output node')
        output_node = partition.output_nodes[0]
        assert isinstance(output_node, Node)
        output_node_list.append(output_node)
    if len(output_node_list) != len(partition_list):
        raise ValueError('length of output_node_list should equal to length of partition_list')
    return output_node_list