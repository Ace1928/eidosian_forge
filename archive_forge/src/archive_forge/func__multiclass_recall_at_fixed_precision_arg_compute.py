from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.enums import ClassificationTask
def _multiclass_recall_at_fixed_precision_arg_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_classes: int, thresholds: Optional[Tensor], min_precision: float, reduce_fn: Callable=_recall_at_precision) -> Tuple[Tensor, Tensor]:
    precision, recall, thresholds = _multiclass_precision_recall_curve_compute(state, num_classes, thresholds)
    if isinstance(state, Tensor):
        res = [reduce_fn(p, r, thresholds, min_precision) for p, r in zip(precision, recall)]
    else:
        res = [reduce_fn(p, r, t, min_precision) for p, r, t in zip(precision, recall, thresholds)]
    recall = torch.stack([r[0] for r in res])
    thresholds = torch.stack([r[1] for r in res])
    return (recall, thresholds)