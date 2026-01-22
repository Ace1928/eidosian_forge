from typing import List, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _edit_distance
def _cer_compute(errors: Tensor, total: Tensor) -> Tensor:
    """Compute the Character error rate.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of characters over all references

    Returns:
        Character error rate score

    """
    return errors / total