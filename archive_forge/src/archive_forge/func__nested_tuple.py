from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _nested_tuple(nested_list: List) -> Tuple:
    """Construct a nested tuple from a nested list.

    Args:
        nested_list: The nested list to convert to a nested tuple.

    Returns:
        A nested tuple with the same content.

    """
    return tuple(map(_nested_tuple, nested_list)) if isinstance(nested_list, list) else nested_list