from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
from torch import Tensor, nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from .data import DataConsumer
def _find_or_add(self, module: RemoteModule) -> Node:
    try:
        return self._find_node(module)
    except ValueError:
        new_node = Node(module)
        self.nodes.append(new_node)
        return new_node