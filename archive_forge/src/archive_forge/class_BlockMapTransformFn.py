import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class BlockMapTransformFn(MapTransformFn):
    """A block-to-block MapTransformFn."""

    def __init__(self, block_fn: MapTransformCallable[Block, Block]):
        self._block_fn = block_fn
        super().__init__(MapTransformFnDataType.Block, MapTransformFnDataType.Block)

    def __call__(self, input: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
        yield from self._block_fn(input, ctx)

    def __repr__(self) -> str:
        return f'BlockMapTransformFn({self._block_fn})'