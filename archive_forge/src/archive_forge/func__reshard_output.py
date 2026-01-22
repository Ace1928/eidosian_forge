from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
from .sharding_spec import (
from .sharding_plan import (
from .sharder import Sharder
def _reshard_output(module: torch.nn.Module, resharding_spec: ShardingSpec) -> torch.nn.Module:
    """
    Hook a module with output resharding in the forward pass according
    to the given ``resharding_spec``.

    Args:
        module (:class:`torch.nn.Module`): Module whose output needs to be resharded.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            The specification describing how the output of the module will be resharded.

    Returns:
        A :class:`torch.nn.Module` object with reshard API hooked.
    """

    def hook_func(_module, _input, output):
        if isinstance(output, ShardedTensor):
            return output.reshard(resharding_spec)
        return output
    module.register_forward_hook(hook_func)
    return module