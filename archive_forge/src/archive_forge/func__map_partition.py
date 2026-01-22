import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@staticmethod
def _map_partition(map_fn, idx: int, block: Block, output_num_blocks: int, schedule: _MergeTaskSchedule, *map_args: List[Any]) -> List[Union[BlockMetadata, Block]]:
    mapper_outputs = map_fn(idx, block, output_num_blocks, *map_args)
    meta = mapper_outputs.pop(-1)
    parts = []
    merge_idx = 0
    while mapper_outputs:
        partition_size = schedule.get_num_reducers_per_merge_idx(merge_idx)
        parts.append(mapper_outputs[:partition_size])
        mapper_outputs = mapper_outputs[partition_size:]
        merge_idx += 1
    assert len(parts) == schedule.num_merge_tasks_per_round, (len(parts), schedule.num_merge_tasks_per_round)
    return parts + [meta]