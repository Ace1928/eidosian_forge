import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
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