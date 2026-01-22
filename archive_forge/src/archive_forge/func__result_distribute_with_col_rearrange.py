import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import (
def _result_distribute_with_col_rearrange(results, input, world_size, weight, pg):
    """
    For col-wise sharding of weight, we need to distribute
    results to each rank. We do them in this function.
    Note that, if the index in the Sharding Spec is not equal to
    the rank number, we need to do the rearrangement based on the
    order given by the Sharding Spec (placement).

    Args:
        results: results from ops applied to inputs from all ranks.
            We need to distribute them back to their original ranks.
        input: tensor to be applied op to.
        world_size: number of ranks.
        weight: sharded weight tensor.
        pg: process group.

    Return: column rearranged result.
    """
    sharding_dim = weight._sharding_spec.dim
    sharding_dim_size = weight.size(sharding_dim)
    dims = list(results[0].size())
    dims[0] = sharding_dim_size
    combined_results = torch.cat(results)
    output = torch.empty(*dims, device=combined_results.device, dtype=combined_results.dtype)
    split_size = get_split_size(sharding_dim_size, world_size)
    output_split_sizes = [0] * world_size
    for idx, placement in enumerate(weight._sharding_spec.placements):
        output_split_sizes[placement.rank()] = get_chunked_dim_size(sharding_dim_size, split_size, idx)
    output = all_to_all_single(output, combined_results, output_split_sizes=output_split_sizes, group=pg)
    rearrange_columns = any((idx != placement.rank() for idx, placement in enumerate(weight._sharding_spec.placements)))
    if not rearrange_columns:
        return output
    indices = []
    for placement in weight._sharding_spec.placements:
        dim_size = output_split_sizes[placement.rank()]
        start = sum([split_size if i < placement.rank() else 0 for i, split_size in enumerate(output_split_sizes)])
        indices += list(range(start, start + dim_size))
    return output.index_select(0, torch.tensor(indices, device=output.device))