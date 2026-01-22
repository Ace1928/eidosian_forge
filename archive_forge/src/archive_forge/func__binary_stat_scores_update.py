from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _binary_stat_scores_update(preds: Tensor, target: Tensor, multidim_average: Literal['global', 'samplewise']='global') -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the statistics."""
    sum_dim = [0, 1] if multidim_average == 'global' else [1]
    tp = ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    fn = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    fp = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    tn = ((target == preds) & (target == 0)).sum(sum_dim).squeeze()
    return (tp, fp, tn, fn)