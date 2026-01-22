from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
class PartitionerConfig(NamedTuple):
    devices: List[Device]
    mode: PartitionMode = PartitionMode.size_based
    transfer_rate_bytes_per_sec: float = 0.0
    node_to_latency_mapping: Dict[Node, NodeLatency] = {}
    node_to_partition_mapping: Dict[Node, int] = {}
    partition_to_logical_device_mapping: Dict[int, List[int]] = {}
    saturate_host: bool = False