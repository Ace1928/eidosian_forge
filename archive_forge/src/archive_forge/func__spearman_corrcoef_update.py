from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
def _spearman_corrcoef_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Spearman Correlation Coefficient.

    Check for same shape and type of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    """
    if not (preds.is_floating_point() and target.is_floating_point()):
        raise TypeError('Expected `preds` and `target` both to be floating point tensors, but got {pred.dtype} and {target.dtype}')
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    return (preds, target)