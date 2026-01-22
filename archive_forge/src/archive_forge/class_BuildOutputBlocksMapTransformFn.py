import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class BuildOutputBlocksMapTransformFn(MapTransformFn):
    """A MapTransformFn that converts UDF-returned data to output blocks."""

    def __init__(self, input_type: MapTransformFnDataType):
        """
        Args:
            input_type: the type of input data.
        """
        self._input_type = input_type
        super().__init__(input_type, MapTransformFnDataType.Block)

    def __call__(self, iter: Iterable[MapTransformFnData], _: TaskContext) -> Iterable[Block]:
        """Convert UDF-returned data to output blocks.

        Args:
            iter: the iterable of UDF-returned data, whose type
                must match self._input_type.
        """
        assert self._target_max_block_size is not None, 'target_max_block_size must be set before running'
        output_buffer = BlockOutputBuffer(self._target_max_block_size)
        if self._input_type == MapTransformFnDataType.Block:
            add_fn = output_buffer.add_block
        elif self._input_type == MapTransformFnDataType.Batch:
            add_fn = output_buffer.add_batch
        else:
            assert self._input_type == MapTransformFnDataType.Row
            add_fn = output_buffer.add
        for data in iter:
            add_fn(data)
            while output_buffer.has_next():
                yield output_buffer.next()
        output_buffer.finalize()
        while output_buffer.has_next():
            yield output_buffer.next()

    @classmethod
    def for_rows(cls) -> 'BuildOutputBlocksMapTransformFn':
        """Return a BuildOutputBlocksMapTransformFn for row input."""
        return cls(MapTransformFnDataType.Row)

    @classmethod
    def for_batches(cls) -> 'BuildOutputBlocksMapTransformFn':
        """Return a BuildOutputBlocksMapTransformFn for batch input."""
        return cls(MapTransformFnDataType.Batch)

    @classmethod
    def for_blocks(cls) -> 'BuildOutputBlocksMapTransformFn':
        """Return a BuildOutputBlocksMapTransformFn for block input."""
        return cls(MapTransformFnDataType.Block)

    def __repr__(self) -> str:
        return f'BuildOutputBlocksMapTransformFn(input_type={self._input_type})'