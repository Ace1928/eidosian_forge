import bisect
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional
import numpy as np
import ray
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockAccessor
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
@ray.remote(num_cpus=0)
class _RandomAccessWorker:

    def __init__(self, key_field):
        self.blocks = None
        self.key_field = key_field
        self.num_accesses = 0
        self.total_time = 0

    def assign_blocks(self, block_ref_dict):
        self.blocks = {k: ray.get(ref) for k, ref in block_ref_dict.items()}

    def get(self, block_index, key):
        start = time.perf_counter()
        result = self._get(block_index, key)
        self.total_time += time.perf_counter() - start
        self.num_accesses += 1
        return result

    def multiget(self, block_indices, keys):
        start = time.perf_counter()
        block = self.blocks[block_indices[0]]
        if len(set(block_indices)) == 1 and isinstance(self.blocks[block_indices[0]], pa.Table):
            block = self.blocks[block_indices[0]]
            col = block[self.key_field]
            indices = np.searchsorted(col, keys)
            acc = BlockAccessor.for_block(block)
            result = [acc._get_row(i) for i in indices]
        else:
            result = [self._get(i, k) for i, k in zip(block_indices, keys)]
        self.total_time += time.perf_counter() - start
        self.num_accesses += 1
        return result

    def ping(self):
        return ray.get_runtime_context().get_node_id()

    def stats(self) -> dict:
        return {'num_blocks': len(self.blocks), 'num_accesses': self.num_accesses, 'total_time': self.total_time}

    def _get(self, block_index, key):
        if block_index is None:
            return None
        block = self.blocks[block_index]
        column = block[self.key_field]
        if isinstance(block, pa.Table):
            column = _ArrowListWrapper(column)
        i = _binary_search_find(column, key)
        if i is None:
            return None
        acc = BlockAccessor.for_block(block)
        return acc._get_row(i)