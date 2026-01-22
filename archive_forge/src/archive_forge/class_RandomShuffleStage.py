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
class RandomShuffleStage(AllToAllStage):
    """Implementation of `Dataset.random_shuffle()`."""

    def __init__(self, seed: Optional[int], output_num_blocks: Optional[int], remote_args: Optional[Dict[str, Any]]=None):

        def do_shuffle(block_list, ctx: TaskContext, clear_input_blocks: bool, block_udf, remote_args):
            num_blocks = block_list.executed_num_blocks()
            if num_blocks == 0:
                return (block_list, {})
            if clear_input_blocks:
                blocks = block_list.copy()
                block_list.clear()
            else:
                blocks = block_list
            context = DataContext.get_current()
            if context.use_push_based_shuffle:
                if output_num_blocks is not None:
                    raise NotImplementedError("Push-based shuffle doesn't support setting num_blocks yet.")
                shuffle_op_cls = PushBasedShufflePartitionOp
            else:
                shuffle_op_cls = SimpleShufflePartitionOp
            random_shuffle_op = shuffle_op_cls(block_udf, random_shuffle=True, random_seed=seed)
            return random_shuffle_op.execute(blocks, output_num_blocks or num_blocks, clear_input_blocks, map_ray_remote_args=remote_args, reduce_ray_remote_args=remote_args, ctx=ctx)
        super().__init__('RandomShuffle', output_num_blocks, do_shuffle, supports_block_udf=True, remote_args=remote_args, sub_stage_names=['ShuffleMap', 'ShuffleReduce'])