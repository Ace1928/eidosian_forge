from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _get_ties(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Get a total number of ties and staistics for p-value calculation for  a given sequence."""
    ties = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    ties_p1 = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    ties_p2 = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    for dim in range(x.shape[1]):
        n_ties = _bincount(x[:, dim])
        n_ties = n_ties[n_ties > 1]
        ties[dim] = (n_ties * (n_ties - 1) // 2).sum()
        ties_p1[dim] = (n_ties * (n_ties - 1.0) * (n_ties - 2)).sum()
        ties_p2[dim] = (n_ties * (n_ties - 1.0) * (2 * n_ties + 5)).sum()
    return (ties, ties_p1, ties_p2)