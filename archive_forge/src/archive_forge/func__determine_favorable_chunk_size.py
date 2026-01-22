import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
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