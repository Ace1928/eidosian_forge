from typing import Any, Dict, List, Optional, Tuple
import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.split import _split_at_indices
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef

    The split (non-shuffle) repartition scheduler.

    First, we calculate global splits needed to produce `output_num_blocks` blocks.
    After the split blocks are generated accordingly, reduce tasks are scheduled
    to combine split blocks together.
    