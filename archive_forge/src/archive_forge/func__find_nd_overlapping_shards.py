from typing import List, Optional, Tuple
from torch.distributed._shard.metadata import ShardMetadata
def _find_nd_overlapping_shards(shards: List[ShardMetadata], sharded_dims: List[int]) -> Optional[Tuple[int, int]]:
    shard_intervals = [[(s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1) for dim in sharded_dims] for s in shards]
    for i in range(len(shards)):
        shard_i = shard_intervals[i]
        for j in range(i + 1, len(shards)):
            shard_j = shard_intervals[j]
            overlap = True
            for interval_i, interval_j in zip(shard_i, shard_j):
                if interval_i[0] > interval_j[1] or interval_j[0] > interval_i[1]:
                    overlap = False
                    break
            if overlap:
                return (i, j)
    return None