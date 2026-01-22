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