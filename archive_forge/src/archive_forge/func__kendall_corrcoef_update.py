from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _kendall_corrcoef_update(preds: Tensor, target: Tensor, concat_preds: Optional[List[Tensor]]=None, concat_target: Optional[List[Tensor]]=None, num_outputs: int=1) -> Tuple[List[Tensor], List[Tensor]]:
    """Update variables required to compute Kendall rank correlation coefficient.

    Args:
        preds: Sequence of data
        target: Sequence of data
        concat_preds: List of batches of preds sequence to be concatenated
        concat_target: List of batches of target sequence to be concatenated
        num_outputs: Number of outputs in multioutput setting

    Raises:
        RuntimeError: If ``preds`` and ``target`` do not have the same shape

    """
    concat_preds = concat_preds or []
    concat_target = concat_target or []
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    if num_outputs == 1:
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)
    concat_preds.append(preds)
    concat_target.append(target)
    return (concat_preds, concat_target)