from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _get_color_areas(inputs: Tensor) -> Dict[Tuple, Tensor]:
    """Measure the size of each instance.

    Args:
        inputs: the input tensor containing the colored pixels.

    Returns:
        A dictionary specifying the `(category_id, instance_id)` and the corresponding number of occurrences.

    """
    unique_keys, unique_keys_area = torch.unique(inputs, dim=0, return_counts=True)
    return dict(zip(_to_tuple(unique_keys), unique_keys_area))