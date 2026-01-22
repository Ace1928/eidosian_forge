import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class MapTransformer:
    """Encapsulates the data transformation logic of a physical MapOperator.

    A MapTransformer may consist of one or more steps, each of which is represented
    as a MapTransformFn. The first MapTransformFn must take blocks as input, and
    the last MapTransformFn must output blocks. The intermediate data types can
    be blocks, rows, or batches.
    """

    def __init__(self, transform_fns: List[MapTransformFn], init_fn: Optional[Callable[[], None]]=None):
        """
        Args:
        transform_fns: A list of `MapTransformFn`s that will be executed sequentially
            to transform data.
        init_fn: A function that will be called before transforming data.
            Used for the actor-based map operator.
        """
        self.set_transform_fns(transform_fns)
        self._init_fn = init_fn if init_fn is not None else lambda: None
        self._target_max_block_size = None

    def set_transform_fns(self, transform_fns: List[MapTransformFn]) -> None:
        """Set the transform functions."""
        assert len(transform_fns) > 0
        assert transform_fns[0].input_type == MapTransformFnDataType.Block, 'The first transform function must take blocks as input.'
        assert transform_fns[-1].output_type == MapTransformFnDataType.Block, 'The last transform function must output blocks.'
        for i in range(len(transform_fns) - 1):
            assert transform_fns[i].output_type == transform_fns[i + 1].input_type, 'The output type of the previous transform function must match the input type of the next transform function.'
        self._transform_fns = transform_fns

    def get_transform_fns(self) -> List[MapTransformFn]:
        """Get the transform functions."""
        return self._transform_fns

    def set_target_max_block_size(self, target_max_block_size: int):
        self._target_max_block_size = target_max_block_size

    def init(self) -> None:
        """Initialize the transformer.

        Should be called before applying the transform.
        """
        self._init_fn()

    def apply_transform(self, input_blocks: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
        """Apply the transform functions to the input blocks."""
        assert self._target_max_block_size is not None, 'target_max_block_size must be set before running'
        for transform_fn in self._transform_fns:
            transform_fn.set_target_max_block_size(self._target_max_block_size)
        iter = input_blocks
        for transform_fn in self._transform_fns:
            iter = transform_fn(iter, ctx)
        return iter

    def fuse(self, other: 'MapTransformer') -> 'MapTransformer':
        """Fuse two `MapTransformer`s together."""
        assert self._target_max_block_size == other._target_max_block_size or (self._target_max_block_size is None or other._target_max_block_size is None)
        target_max_block_size = self._target_max_block_size or other._target_max_block_size
        self_init_fn = self._init_fn
        other_init_fn = other._init_fn

        def fused_init_fn():
            self_init_fn()
            other_init_fn()
        fused_transform_fns = self._transform_fns + other._transform_fns
        transformer = MapTransformer(fused_transform_fns, init_fn=fused_init_fn)
        transformer.set_target_max_block_size(target_max_block_size)
        return transformer