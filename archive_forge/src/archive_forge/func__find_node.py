from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
from torch import Tensor, nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from .data import DataConsumer
def _find_node(self, module: RemoteModule) -> Node:
    for n in self.nodes:
        if n.module is module:
            return n
    raise ValueError