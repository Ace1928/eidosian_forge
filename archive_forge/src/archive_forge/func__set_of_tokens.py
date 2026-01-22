import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
@staticmethod
def _set_of_tokens(input_ids: Tensor) -> Set:
    """Return set of tokens from the `input_ids` :class:`~torch.Tensor`."""
    return set(input_ids.tolist())