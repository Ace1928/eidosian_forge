from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _stat_scores_compute(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> Tensor:
    """Compute the number of true positives, false positives, true negatives, false negatives.

    Concatenates the input tensors along with the support into one output.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives

    """
    stats = [tp.unsqueeze(-1), fp.unsqueeze(-1), tn.unsqueeze(-1), fn.unsqueeze(-1), tp.unsqueeze(-1) + fn.unsqueeze(-1)]
    outputs: Tensor = torch.cat(stats, -1)
    return torch.where(outputs < 0, tensor(-1, device=outputs.device), outputs)