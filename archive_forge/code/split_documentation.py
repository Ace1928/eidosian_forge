import itertools
import logging
from typing import Iterable, List, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.types import ObjectRef
Split blocks at the provided index.
    Args:
        blocks_with_metadata: Block futures to split, including the associated metadata.
        index: The (global) index at which to split the blocks.
    Returns:
        The block split futures and their metadata for left and right of the index.
    