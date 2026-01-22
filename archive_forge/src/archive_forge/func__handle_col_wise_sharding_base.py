import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import (
def _handle_col_wise_sharding_base(op_func, col_dim, input, world_size, weight, local_shard, pg, gathered_inputs, mode=None, gathered_per_sample_weights=None, gathered_offsets=None, padding_idx=None):
    """
    For col-wise sharding of weight, lots of logic are common.
    So we extract the common logic and put in this function:
    Step 1. To get input from each rank and
    Step 2. To perform the op on the concatenated tensor.
    Step 3. To distribute results to each rank with col rearrangement.
    Step 4. To concatenate all results from all ranks.

    Args:
        op_func: operator which is applied to the input tensor.
        col_dim: dim of result tensor after the operation.
        input: tensor to be applied op on.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise sharded weight tensor.
        pg: process group.
        gathered_inputs: list of inputs from all ranks. If specified, we
            don't need to communicate with each rank any more.
        mode: aggregation mode of EmbeddingBag.
        gathered_per_sample_weights: per_sample_weights across all ranks.
        gathered_offsets: offsets across all ranks.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”.
            Note that the embedding vector at padding_idx is
            excluded from the reduction.

    Return: final result of input being applied with the op.
    """
    results = []
    for i, inp in enumerate(gathered_inputs):
        if op_func == torch.nn.functional.embedding_bag:
            result = op_func(inp, local_shard, offsets=gathered_offsets[i] if gathered_offsets is not None else None, mode=mode, per_sample_weights=gathered_per_sample_weights[i] if gathered_per_sample_weights is not None else None, padding_idx=padding_idx)
        elif op_func == torch.nn.functional.embedding:
            result = op_func(inp, local_shard, padding_idx=padding_idx)
        else:
            result = op_func(inp, local_shard)
        results.append(torch.transpose(result, 0, col_dim))
    output = _result_distribute_with_col_rearrange(results, input, world_size, weight, pg)
    return torch.transpose(output, 0, col_dim)