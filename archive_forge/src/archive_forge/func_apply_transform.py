import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
def apply_transform(self, input_blocks: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
    """Apply the transform functions to the input blocks."""
    assert self._target_max_block_size is not None, 'target_max_block_size must be set before running'
    for transform_fn in self._transform_fns:
        transform_fn.set_target_max_block_size(self._target_max_block_size)
    iter = input_blocks
    for transform_fn in self._transform_fns:
        iter = transform_fn(iter, ctx)
    return iter