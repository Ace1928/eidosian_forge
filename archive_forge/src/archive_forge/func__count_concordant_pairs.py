from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _count_concordant_pairs(preds: Tensor, target: Tensor) -> Tensor:
    """Count a total number of concordant pairs in given sequences."""
    return torch.cat([_concordant_element_sum(preds, target, i) for i in range(preds.shape[0])]).sum(0)