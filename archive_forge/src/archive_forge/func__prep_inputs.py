import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
def _prep_inputs(t: torch.Tensor) -> torch.Tensor:
    if not low_mem:
        if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        t = t.reshape(-1, *t.shape[no_batch_dims:])
    else:
        t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
    return t