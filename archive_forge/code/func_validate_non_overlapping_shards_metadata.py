from typing import List, Optional, Tuple
from torch.distributed._shard.metadata import ShardMetadata
def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    """
    Ensures none of the shards overlap with each other.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard.
    Raises:
        ``ValueError`` if there's overlap in any two shards.
    """
    if not shards or len(shards) == 1:
        return
    sharded_dims: List[int] = []
    for dim in range(len(shards[0].shard_offsets)):
        for i in range(1, len(shards)):
            if shards[i].shard_offsets[dim] != shards[0].shard_offsets[dim] or shards[i].shard_sizes[dim] != shards[0].shard_sizes[dim]:
                sharded_dims.append(dim)
                break
    pair: Optional[Tuple[int, int]] = None
    if len(sharded_dims) == 0:
        pair = (0, 1)
    elif len(sharded_dims) == 1:
        pair = _find_1d_overlapping_shards(shards, sharded_dims[0])
    else:
        pair = _find_nd_overlapping_shards(shards, sharded_dims)
    if pair:
        raise ValueError(f'Shards {shards[pair[0]]} and {shards[pair[1]]} overlap')