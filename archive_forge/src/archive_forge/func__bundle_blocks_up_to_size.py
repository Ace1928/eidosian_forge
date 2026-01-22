import collections
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
def _bundle_blocks_up_to_size(blocks: List[Tuple[ObjectRef[Block], BlockMetadata]], target_size: int) -> List[Tuple[Tuple[ObjectRef[Block]], Tuple[BlockMetadata]]]:
    """Group blocks into bundles that are up to (but not exceeding) the provided target
    size.
    """
    block_bundles: List[List[Tuple[ObjectRef[Block], BlockMetadata]]] = []
    curr_bundle: List[Tuple[ObjectRef[Block], BlockMetadata]] = []
    curr_bundle_size = 0
    for b, m in blocks:
        num_rows = m.num_rows
        if num_rows is None:
            num_rows = float('inf')
        if curr_bundle_size > 0 and curr_bundle_size + num_rows > target_size:
            block_bundles.append(curr_bundle)
            curr_bundle = []
            curr_bundle_size = 0
        curr_bundle.append((b, m))
        curr_bundle_size += num_rows
    if curr_bundle:
        block_bundles.append(curr_bundle)
    if len(blocks) / len(block_bundles) >= 10:
        logger.warning(f'`batch_size` is set to {target_size}, which reduces parallelism from {len(blocks)} to {len(block_bundles)}. If the performance is worse than expected, this may indicate that the batch size is too large or the input block size is too small. To reduce batch size, consider decreasing `batch_size` or use the default in `map_batches`. To increase input block size, consider decreasing `parallelism` in read.')
    return [tuple(zip(*block_bundle)) for block_bundle in block_bundles]