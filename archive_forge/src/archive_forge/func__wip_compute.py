from typing import List, Tuple, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _edit_distance
def _wip_compute(errors: Tensor, target_total: Tensor, preds_total: Tensor) -> Tensor:
    """Compute the Word Information Preserved.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        target_total: Number of words overall references
        preds_total: Number of words overall prediction

    Returns:
        Word Information Preserved score

    """
    return errors / target_total * (errors / preds_total)