from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix
from torchmetrics.utilities.exceptions import TorchMetricsUserError
def _pairwise_minkowski_distance_update(x: Tensor, y: Optional[Tensor]=None, exponent: float=2, zero_diagonal: Optional[bool]=None) -> Tensor:
    """Calculate the pairwise minkowski distance matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
        exponent: int or float larger than 1, exponent to which the difference between preds and target is to be raised
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
    if not (isinstance(exponent, (float, int)) and exponent >= 1):
        raise TorchMetricsUserError(f'Argument ``p`` must be a float or int greater than 1, but got {exponent}')
    _orig_dtype = x.dtype
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    distance = (x.unsqueeze(1) - y.unsqueeze(0)).abs().pow(exponent).sum(-1).pow(1.0 / exponent)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance.to(_orig_dtype)