from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _binary_confusion_matrix_update(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = (target * 2 + preds).to(torch.long)
    bins = _bincount(unique_mapping, minlength=4)
    return bins.reshape(2, 2)