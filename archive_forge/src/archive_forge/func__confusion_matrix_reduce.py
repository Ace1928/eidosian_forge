from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _confusion_matrix_reduce(confmat: Tensor, normalize: Optional[Literal['true', 'pred', 'all', 'none']]=None) -> Tensor:
    """Reduce an un-normalized confusion matrix.

    Args:
        confmat: un-normalized confusion matrix
        normalize: normalization method.
            - `"true"` will divide by the sum of the column dimension.
            - `"pred"` will divide by the sum of the row dimension.
            - `"all"` will divide by the sum of the full matrix
            - `"none"` or `None` will apply no reduction.

    Returns:
        Normalized confusion matrix

    """
    allowed_normalize = ('true', 'pred', 'all', 'none', None)
    if normalize not in allowed_normalize:
        raise ValueError(f'Argument `normalize` needs to one of the following: {allowed_normalize}')
    if normalize is not None and normalize != 'none':
        confmat = confmat.float() if not confmat.is_floating_point() else confmat
        if normalize == 'true':
            confmat = confmat / confmat.sum(dim=-1, keepdim=True)
        elif normalize == 'pred':
            confmat = confmat / confmat.sum(dim=-2, keepdim=True)
        elif normalize == 'all':
            confmat = confmat / confmat.sum(dim=[-2, -1], keepdim=True)
        nan_elements = confmat[torch.isnan(confmat)].nelement()
        if nan_elements:
            confmat[torch.isnan(confmat)] = 0
            rank_zero_warn(f'{nan_elements} NaN values found in confusion matrix have been replaced with zeros.')
    return confmat