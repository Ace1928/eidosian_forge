from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
def _unsqueeze_tensors(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    if preds.ndim == 2:
        return (preds, target)
    return (preds.unsqueeze(1), target.unsqueeze(1))