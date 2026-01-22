from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _get_void_color(things: Set[int], stuffs: Set[int]) -> Tuple[int, int]:
    """Get an unused color ID.

    Args:
        things: The set of category IDs for things.
        stuffs: The set of category IDs for stuffs.

    Returns:
        A new color ID that does not belong to things nor stuffs.

    """
    unused_category_id = 1 + max([0, *list(things), *list(stuffs)])
    return (unused_category_id, 0)