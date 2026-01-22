from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _convert_sequence_to_dense_rank(x: Tensor, sort: bool=False) -> Tensor:
    """Convert a sequence to the rank tensor."""
    if sort:
        x = x.sort(dim=0).values
    _ones = torch.zeros(1, x.shape[1], dtype=torch.int32, device=x.device)
    return _cumsum(torch.cat([_ones, (x[1:] != x[:-1]).int()], dim=0), dim=0)