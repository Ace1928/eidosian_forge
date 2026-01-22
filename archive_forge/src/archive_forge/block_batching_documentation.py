import collections
import itertools
from contextlib import nullcontext
from typing import Any, Callable, Iterator, Optional, TypeVar
import ray
from ray.data._internal.block_batching.interfaces import BlockPrefetcher
from ray.data._internal.block_batching.util import (
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats
from ray.data.block import Block, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
Given an iterable of Block Object References, returns an iterator
    over these object reference while prefetching `num_block_to_prefetch`
    blocks in advance.

    Args:
        block_ref_iter: An iterator over block object references.
        num_blocks_to_prefetch: The number of blocks to prefetch ahead of the
            current block during the scan.
        stats: Dataset stats object used to store block wait time.
    