import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def try_swap_nodes(n0, n1, p0, p1, node_to_latency_mapping, transfer_rate_per_sec):
    cost = float('inf')
    swap_nodes(n0, n1, p0, p1)
    reorganize_partitions(self.partitions)
    if not check_dependency(p0) and (not check_dependency(p1)):
        reset_partition_device(self.partitions)
        partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
        found_device = get_device_to_partitions_mapping(self.partitions, self.devices)
        if not found_device:
            cost = float('inf')
        else:
            cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
    swap_nodes(n1, n0, p0, p1)
    reorganize_partitions(self.partitions)
    reset_partition_device(self.partitions)
    get_device_to_partitions_mapping(self.partitions, self.devices)
    return cost