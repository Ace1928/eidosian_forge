from typing import Any, Dict, List, Optional, Tuple, Union
from ray.data._internal.block_list import BlockList
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import Block, BlockMetadata
class ShuffleOp:
    """
    A generic shuffle operator. Callers should first implement the `map` and
    `reduce` static methods then choose a plan to execute the shuffle by
    inheriting from the appropriate class. A SimpleShufflePlan is provided
    below. Any custom arguments for map and reduce tasks should be specified by
    setting `ShuffleOp._map_args` and `ShuffleOp._reduce_args`.
    """

    def __init__(self, map_args: List[Any]=None, reduce_args: List[Any]=None):
        self._map_args = map_args or []
        self._reduce_args = reduce_args or []
        assert isinstance(self._map_args, list)
        assert isinstance(self._reduce_args, list)

    @staticmethod
    def map(idx: int, block: Block, output_num_blocks: int, *map_args: List[Any]) -> List[Union[BlockMetadata, Block]]:
        """
        Map function to be run on each input block.

        Returns list of [BlockMetadata, O1, O2, O3, ...output_num_blocks].
        """
        raise NotImplementedError

    @staticmethod
    def reduce(*mapper_outputs: List[Block], partial_reduce: bool=False) -> (Block, BlockMetadata):
        """
        Reduce function to be run for each output block.

        Args:
            mapper_outputs: List of blocks to reduce.
            partial_reduce: A flag passed by the shuffle operator that
                indicates whether we should partially or fully reduce the
                mapper outputs.

        Returns:
            The reduced block and its metadata.
        """
        raise NotImplementedError