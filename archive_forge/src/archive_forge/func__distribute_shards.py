from typing import List
import numpy as np
def _distribute_shards(num_shards: int, max_num_jobs: int) -> List[range]:
    """
    Get the range of shard indices per job.
    If num_shards<max_num_jobs, then num_shards jobs are given a range of one shard.
    The shards indices order is preserved: e.g. all the first shards are given the first job.
    Moreover all the jobs are given approximately the same number of shards.

    Example:

    ```python
    >>> _distribute_shards(2, max_num_jobs=4)
    [range(0, 1), range(1, 2)]
    >>> _distribute_shards(10, max_num_jobs=3)
    [range(0, 4), range(4, 7), range(7, 10)]
    ```
    """
    shards_indices_per_group = []
    for group_idx in range(max_num_jobs):
        num_shards_to_add = num_shards // max_num_jobs + (group_idx < num_shards % max_num_jobs)
        if num_shards_to_add == 0:
            break
        start = shards_indices_per_group[-1].stop if shards_indices_per_group else 0
        shard_indices = range(start, start + num_shards_to_add)
        shards_indices_per_group.append(shard_indices)
    return shards_indices_per_group