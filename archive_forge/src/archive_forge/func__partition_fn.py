from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from functools import partial
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor, distribute_module
def _partition_fn(self, name, module, device_mesh):
    if isinstance(module, nn.Linear):
        module.register_parameter('weight', nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(1)])))
        if module.bias is not None:
            module.register_parameter('bias', nn.Parameter(distribute_tensor(module.bias, device_mesh, [Replicate()])))
    else:
        raise NotImplementedError('RowwiseParallel currently only support nn.Linear!')