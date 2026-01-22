import collections
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import ray
from ray.data._internal.block_batching.interfaces import Batch, BlockPrefetcher
from ray.data._internal.block_batching.util import (
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import make_async_gen
from ray.data.block import Block, BlockMetadata, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
def _async_iter_batches(block_refs: Iterator[Tuple[ObjectRef[Block], BlockMetadata]]) -> Iterator[DataBatch]:
    block_refs = prefetch_batches_locally(block_ref_iter=block_refs, prefetcher=prefetcher, num_batches_to_prefetch=prefetch_batches, batch_size=batch_size, eager_free=eager_free)
    block_iter = resolve_block_refs(block_ref_iter=block_refs, stats=stats)
    batch_iter = blocks_to_batches(block_iter=block_iter, stats=stats, batch_size=batch_size, drop_last=drop_last, shuffle_buffer_min_size=shuffle_buffer_min_size, shuffle_seed=shuffle_seed, ensure_copy=ensure_copy)
    batch_iter = _format_in_threadpool(batch_iter, stats=stats, batch_format=batch_format, collate_fn=collate_fn, num_threadpool_workers=prefetch_batches)
    if finalize_fn is not None:
        batch_iter = finalize_batches(batch_iter, finalize_fn=finalize_fn, stats=stats)
    batch_iter: Iterator[Batch] = restore_original_order(batch_iter)
    yield from extract_data_from_batch(batch_iter)