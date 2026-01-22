import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
class ChunkSizeTuner:

    def __init__(self, max_chunk_size: int=512):
        self.max_chunk_size = max_chunk_size
        self.cached_chunk_size: Optional[int] = None
        self.cached_arg_data: Optional[tuple] = None

    def _determine_favorable_chunk_size(self, fn: Callable, args: tuple, min_chunk_size: int) -> int:
        logging.info('Tuning chunk size...')
        if min_chunk_size >= self.max_chunk_size:
            return min_chunk_size
        candidates: List[int] = [2 ** l for l in range(int(math.log(self.max_chunk_size, 2)) + 1)]
        candidates = [c for c in candidates if c > min_chunk_size]
        candidates = [min_chunk_size] + candidates
        candidates[-1] += 4

        def test_chunk_size(chunk_size: int) -> bool:
            try:
                with torch.no_grad():
                    fn(*args, chunk_size=chunk_size)
                return True
            except RuntimeError:
                return False
        min_viable_chunk_size_index = 0
        i = len(candidates) - 1
        while i > min_viable_chunk_size_index:
            viable = test_chunk_size(candidates[i])
            if not viable:
                i = (min_viable_chunk_size_index + i) // 2
            else:
                min_viable_chunk_size_index = i
                i = (i + len(candidates) - 1) // 2
        return candidates[min_viable_chunk_size_index]

    def _compare_arg_caches(self, ac1: Iterable, ac2: Iterable) -> bool:
        consistent = True
        for a1, a2 in zip(ac1, ac2):
            assert type(ac1) == type(ac2)
            if isinstance(ac1, (list, tuple)):
                consistent &= self._compare_arg_caches(a1, a2)
            elif isinstance(ac1, dict):
                a1_items = [v for _, v in sorted(a1.items(), key=lambda x: x[0])]
                a2_items = [v for _, v in sorted(a2.items(), key=lambda x: x[0])]
                consistent &= self._compare_arg_caches(a1_items, a2_items)
            else:
                consistent &= a1 == a2
        return consistent

    def tune_chunk_size(self, representative_fn: Callable, args: tuple, min_chunk_size: int) -> int:
        consistent = True
        arg_data: tuple = tree_map(lambda a: a.shape if isinstance(a, torch.Tensor) else a, args, object)
        if self.cached_arg_data is not None:
            assert len(self.cached_arg_data) == len(arg_data)
            consistent = self._compare_arg_caches(self.cached_arg_data, arg_data)
        else:
            consistent = False
        if not consistent:
            self.cached_chunk_size = self._determine_favorable_chunk_size(representative_fn, args, min_chunk_size)
            self.cached_arg_data = arg_data
        assert self.cached_chunk_size is not None
        return self.cached_chunk_size