import copy
import logging
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.execution.interfaces import NodeIdStr, RefBundle
from ray.data._internal.execution.legacy_compat import execute_to_legacy_bundle_iterator
from ray.data._internal.execution.operators.output_splitter import OutputSplitter
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import create_dataset_tag
from ray.data.block import Block, BlockMetadata
from ray.data.iterator import DataIterator
from ray.types import ObjectRef
from ray.util.debug import log_once
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class StreamSplitDataIterator(DataIterator):
    """Implements a collection of iterators over a shared data stream."""

    @staticmethod
    def create(base_dataset: 'Dataset', n: int, equal: bool, locality_hints: Optional[List[NodeIdStr]]) -> List['StreamSplitDataIterator']:
        """Create a split iterator from the given base Dataset and options.

        See also: `Dataset.streaming_split`.
        """
        coord_actor = SplitCoordinator.options(max_concurrency=n, scheduling_strategy=NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)).remote(base_dataset, n, equal, locality_hints)
        return [StreamSplitDataIterator(base_dataset, coord_actor, i, n) for i in range(n)]

    def __init__(self, base_dataset: 'Dataset', coord_actor: ray.actor.ActorHandle, output_split_idx: int, world_size: int):
        self._base_dataset = base_dataset
        self._coord_actor = coord_actor
        self._output_split_idx = output_split_idx
        self._world_size = world_size
        self._iter_stats = DatasetStats(stages={}, parent=None)

    def _to_block_iterator(self) -> Tuple[Iterator[Tuple[ObjectRef[Block], BlockMetadata]], Optional[DatasetStats], bool]:

        def gen_blocks() -> Iterator[Tuple[ObjectRef[Block], BlockMetadata]]:
            cur_epoch = ray.get(self._coord_actor.start_epoch.remote(self._output_split_idx))
            future: ObjectRef[Optional[ObjectRef[Block]]] = self._coord_actor.get.remote(cur_epoch, self._output_split_idx)
            while True:
                block_ref: Optional[Tuple[ObjectRef[Block], BlockMetadata]] = ray.get(future)
                if not block_ref:
                    break
                else:
                    future = self._coord_actor.get.remote(cur_epoch, self._output_split_idx)
                    yield block_ref
        return (gen_blocks(), self._iter_stats, False)

    def stats(self) -> str:
        """Implements DataIterator."""
        summary = ray.get(self._coord_actor.stats.remote())
        summary.iter_stats = self._iter_stats.to_summary().iter_stats
        return summary.to_string()

    def schema(self) -> Union[type, 'pyarrow.lib.Schema']:
        """Implements DataIterator."""
        return self._base_dataset.schema()

    def world_size(self) -> int:
        """Returns the number of splits total."""
        return self._world_size

    def _get_dataset_tag(self):
        return create_dataset_tag(self._base_dataset._plan._dataset_name, self._base_dataset._uuid, self._output_split_idx)