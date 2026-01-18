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
Restores the original order of the provided `batch_iter`

    This function will yield items from `base_iterator` in the correct order based on
    each batch's batch_idx. All indexes are expected to be unique.

    `batch_iter` is expected to not have any missing indexes. All indexes from 0 to len
    (base_iterator) must be present.
    