import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
def _fuse_one_to_one_stages(stages: List[Stage]) -> List[Stage]:
    """Fuses compatible one-to-one stages.

    Args:
        stages: Stages to try to fuse.

    Returns:
        Fused stages.
    """
    fused_stages = []
    prev_stage = None
    for idx, stage in enumerate(stages):
        if prev_stage is None:
            prev_stage = stage
        elif stage.can_fuse(prev_stage):
            prev_stage = stage.fuse(prev_stage)
        else:
            fused_stages.append(prev_stage)
            prev_stage = stage
    if prev_stage:
        fused_stages.append(prev_stage)
        prev_stage = None
    return fused_stages