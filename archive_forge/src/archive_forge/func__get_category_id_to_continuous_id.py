from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _get_category_id_to_continuous_id(things: Set[int], stuffs: Set[int]) -> Dict[int, int]:
    """Convert original IDs to continuous IDs.

    Args:
        things: All unique IDs for things classes.
        stuffs: All unique IDs for stuff classes.

    Returns:
        A mapping from the original category IDs to continuous IDs (i.e., 0, 1, 2, ...).

    """
    thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things)}
    stuff_id_to_continuous_id = {stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuffs)}
    cat_id_to_continuous_id = {}
    cat_id_to_continuous_id.update(thing_id_to_continuous_id)
    cat_id_to_continuous_id.update(stuff_id_to_continuous_id)
    return cat_id_to_continuous_id