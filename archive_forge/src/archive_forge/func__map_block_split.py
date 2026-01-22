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
def _map_block_split(block_fn: BlockTransform, input_files: List[str], fn: Optional[UserDefinedFunction], num_blocks: int, *blocks_and_fn_args: Union[Block, Any], **fn_kwargs) -> BlockPartition:
    stats = BlockExecStats.builder()
    blocks, fn_args = (blocks_and_fn_args[:num_blocks], blocks_and_fn_args[num_blocks:])
    if fn is not None:
        fn_args = (fn,) + fn_args
    new_metas = []
    for new_block in block_fn(blocks, *fn_args, **fn_kwargs):
        accessor = BlockAccessor.for_block(new_block)
        new_meta = BlockMetadata(num_rows=accessor.num_rows(), size_bytes=accessor.size_bytes(), schema=accessor.schema(), input_files=input_files, exec_stats=stats.build())
        yield new_block
        new_metas.append(new_meta)
        stats = BlockExecStats.builder()
    yield new_metas