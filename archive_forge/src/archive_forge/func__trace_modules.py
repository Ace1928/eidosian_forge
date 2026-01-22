from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
from torch import Tensor, nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from .data import DataConsumer
def _trace_modules(self, node: Node) -> List[Node]:
    """Compiles a list of modules (starting from module number module_idx), where each module in the list
        gets the output of previous module in the list as its input. So every module in the list, except the
        first one should have only one input, and similarly, every module in the list, except the last one
        should have only one output.
        """
    partition = []
    current_node = node
    while True:
        partition.append(current_node)
        if len(current_node.output_consumers) != 1:
            break
        if current_node.num_outputs is not None:
            break
        next_node = current_node.output_consumers[0].consumer
        if next_node.inputs != [DataSource(current_node, 0)]:
            break
        if next_node.module.on != current_node.module.on:
            break
        if next_node.module.device != current_node.module.device:
            break
        current_node = next_node
    return partition