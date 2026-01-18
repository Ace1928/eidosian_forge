import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
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