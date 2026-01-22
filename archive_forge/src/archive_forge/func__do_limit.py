import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.fast_repartition import fast_repartition
from ray.data._internal.plan import AllToAllStage
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.shuffle_and_partition import (
from ray.data._internal.sort import SortKey, sort_impl
from ray.data._internal.split import _split_at_index, _split_at_indices
from ray.data.block import (
from ray.data.context import DataContext
def _do_limit(self, input_block_list: BlockList, clear_input_blocks: bool, *_):
    if clear_input_blocks:
        block_list = input_block_list.copy()
        input_block_list.clear()
    else:
        block_list = input_block_list
    block_list = block_list.truncate_by_rows(self._limit)
    blocks, metadata, _, _ = _split_at_index(block_list, self._limit)
    return (BlockList(blocks, metadata, owned_by_consumer=block_list._owned_by_consumer), {})