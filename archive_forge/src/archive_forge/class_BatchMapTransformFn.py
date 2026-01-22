import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class BatchMapTransformFn(MapTransformFn):
    """A batch-to-batch MapTransformFn."""

    def __init__(self, batch_fn: MapTransformCallable[DataBatch, DataBatch]):
        self._batch_fn = batch_fn
        super().__init__(MapTransformFnDataType.Batch, MapTransformFnDataType.Batch)

    def __call__(self, input: Iterable[DataBatch], ctx: TaskContext) -> Iterable[DataBatch]:
        yield from self._batch_fn(input, ctx)

    def __repr__(self) -> str:
        return f'BatchMapTransformFn({self._batch_fn})'