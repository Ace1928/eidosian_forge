from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _discordant_element_sum(x: Tensor, y: Tensor, i: int) -> Tensor:
    """Count a total number of discordant pairs in a single sequences."""
    return torch.logical_or(torch.logical_and(x[i] > x[i + 1:], y[i] < y[i + 1:]), torch.logical_and(x[i] < x[i + 1:], y[i] > y[i + 1:])).sum(0).unsqueeze(0)