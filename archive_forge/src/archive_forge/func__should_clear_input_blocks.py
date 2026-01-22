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
def _should_clear_input_blocks(self, blocks: BlockList, stage_idx: int):
    """Whether the provided blocks should be cleared when passed into the stage.

        Args:
            blocks: The blocks that we may want to clear.
            stage_idx: The position of the stage in the optimized after-snapshot chain.
        """
    if stage_idx != 0 or self._stages_before_snapshot:
        return True
    elif isinstance(blocks, LazyBlockList):
        return True
    else:
        return False