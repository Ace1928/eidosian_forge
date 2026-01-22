import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter
from ._common import (
def _validate_embedding_param(args, kwargs):
    """
    Validate input params of sharded embedding op.

    Args:
        input: list of ID used for lookup.
        weight: sharded weight tensor.
        kwargs: same as normal Embedding.

    Return: None.
    """
    input = args[0]
    weight = args[1]
    max_norm = kwargs.get('max_norm')
    scale_grad_by_freq = kwargs.get('scale_grad_by_freq')
    sparse = kwargs.get('sparse')
    if not isinstance(input, torch.Tensor):
        raise TypeError('input need to be torch.Tensor')
    if not isinstance(weight, ShardedTensor):
        raise TypeError('weight needs to be ShardedTensor')
    weight_size = weight.size()
    if len(weight_size) != 2:
        raise ValueError('Weight needs to have exactly 2 dims')
    if int(torch.min(input).item()) < 0:
        raise ValueError('Index out of range in Input %d %d', int(torch.min(input).item()), weight_size[1])
    if int(torch.max(input).item()) >= weight_size[0]:
        raise ValueError('Index out of range in Input %d %d', int(torch.max(input).item()), weight_size[1])
    if scale_grad_by_freq:
        raise RuntimeError('nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!')
    if sparse:
        raise RuntimeError('nn.Embedding weight sharded with flag on "sparse" not supported!')
    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')
    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError('Only ChunkShardingSpec supported for ShardedTensor ops!')
    if len(weight.local_shards()) != 1:
        raise ValueError('Only one local shard supported!')