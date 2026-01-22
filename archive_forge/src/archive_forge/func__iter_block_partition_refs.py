import math
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats, _get_or_create_stats_actor
from ray.data._internal.util import _split_list
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def _iter_block_partition_refs(self) -> Iterator[Tuple[ObjectRef[MaybeBlockPartition], Union[None, ObjectRef[BlockMetadata]]]]:
    """Iterate over the block futures and their corresponding metadata futures.

        This does NOT block on the execution of each submitted task.
        """
    outer = self

    class Iter:

        def __init__(self):
            self._pos = -1

        def __iter__(self):
            return self

        def __next__(self):
            self._pos += 1
            if self._pos < len(outer._tasks):
                return outer._get_or_compute(self._pos)
            raise StopIteration
    return Iter()