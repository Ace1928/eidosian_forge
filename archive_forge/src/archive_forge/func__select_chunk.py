import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
def _select_chunk(t: torch.Tensor) -> torch.Tensor:
    return t[i:i + chunk_size] if t.shape[0] != 1 else t