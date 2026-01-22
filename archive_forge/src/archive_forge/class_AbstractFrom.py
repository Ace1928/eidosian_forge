import abc
from typing import TYPE_CHECKING, List, Union
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data.block import Block, BlockMetadata
from ray.types import ObjectRef
class AbstractFrom(LogicalOperator, metaclass=abc.ABCMeta):
    """Abstract logical operator for `from_*`."""

    def __init__(self, input_blocks: List[ObjectRef[Block]], input_metadata: List[BlockMetadata]):
        super().__init__(self.__class__.__name__, [])
        assert len(input_blocks) == len(input_metadata), (len(input_blocks), len(input_metadata))
        self._input_data = [RefBundle([(input_blocks[i], input_metadata[i])], owns_blocks=False) for i in range(len(input_blocks))]

    @property
    def input_data(self) -> List[RefBundle]:
        return self._input_data