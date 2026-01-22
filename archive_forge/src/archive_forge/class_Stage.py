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
class Stage:
    """Represents a Dataset transform stage (e.g., map or shuffle)."""

    def __init__(self, name: str, num_blocks: Optional[int]):
        self.name = name
        self.num_blocks = num_blocks

    def __call__(self, blocks: BlockList, clear_input_blocks: bool) -> Tuple[BlockList, dict]:
        """Execute this stage against the given blocks."""
        raise NotImplementedError

    def can_fuse(self, other: 'Stage') -> bool:
        """Return whether this can be fused with another stage."""
        raise NotImplementedError

    def fuse(self, other: 'Stage') -> 'Stage':
        """Fuse this stage with a compatible stage."""
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}("{self.name}")'

    def __str__(self):
        return repr(self)