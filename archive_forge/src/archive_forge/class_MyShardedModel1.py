import copy
import random
import torch
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import (
class MyShardedModel1(torch.nn.Module):

    def __init__(self, spec=None, group=None, init_rrefs=True) -> None:
        super().__init__()
        if spec is not None:
            self.sharded_tensor1 = sharded_tensor.rand(spec, 10, 20, process_group=group, init_rrefs=init_rrefs)
        else:
            self.sharded_tensor1 = None
        self.random_tensor1 = torch.nn.Parameter(torch.rand(2, 2))
        self.submodule = MyShardedModel2(spec, group, init_rrefs)