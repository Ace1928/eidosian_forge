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
def truncate_by_rows(self, limit: int) -> 'LazyBlockList':
    """Truncate the block list to the minimum number of blocks that contains at
        least limit rows.

        If the number of rows is not available, it will be treated as a 0-row block and
        will be included in the truncated output.
        """
    self._check_if_cleared()
    out_tasks, out_blocks, out_blocks_meta, out_cached_meta = ([], [], [], [])
    out_num_rows = 0
    for t, b, bm, c in zip(self._tasks, self._block_partition_refs, self._block_partition_meta_refs, self._cached_metadata):
        m = t.get_metadata()
        num_rows = m.num_rows
        if num_rows is None:
            num_rows = 0
        out_tasks.append(t)
        out_blocks.append(b)
        out_blocks_meta.append(bm)
        out_cached_meta.append(c)
        out_num_rows += num_rows
        if out_num_rows >= limit:
            break
    return LazyBlockList(out_tasks, out_blocks, out_blocks_meta, out_cached_meta, owned_by_consumer=self._owned_by_consumer)