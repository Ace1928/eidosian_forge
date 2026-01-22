import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class MapTransformFn:
    """Represents a single transform function in a MapTransformer."""

    def __init__(self, input_type: MapTransformFnDataType, output_type: MapTransformFnDataType):
        """
        Args:
            callable: the underlying Python callable object.
            input_type: the type of the input data.
            output_type: the type of the output data.
        """
        self._callable = callable
        self._input_type = input_type
        self._output_type = output_type
        self._target_max_block_size = None

    @abstractmethod
    def __call__(self, input: Iterable[MapTransformFnData], ctx: TaskContext) -> Iterable[MapTransformFnData]:
        ...

    @property
    def input_type(self) -> MapTransformFnDataType:
        return self._input_type

    @property
    def output_type(self) -> MapTransformFnDataType:
        return self._output_type

    def set_target_max_block_size(self, target_max_block_size: int):
        self._target_max_block_size = target_max_block_size