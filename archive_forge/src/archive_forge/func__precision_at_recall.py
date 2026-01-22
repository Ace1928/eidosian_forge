from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.recall_fixed_precision import (
from torchmetrics.utilities.enums import ClassificationTask
def _precision_at_recall(precision: Tensor, recall: Tensor, thresholds: Tensor, min_recall: float) -> Tuple[Tensor, Tensor]:
    try:
        max_precision, _, best_threshold = max(((p, r, t) for p, r, t in zip(precision, recall, thresholds) if r >= min_recall))
    except ValueError:
        max_precision = torch.tensor(0.0, device=precision.device, dtype=precision.dtype)
        best_threshold = torch.tensor(0)
    if max_precision == 0.0:
        best_threshold = torch.tensor(1000000.0, device=thresholds.device, dtype=thresholds.dtype)
    return (max_precision, best_threshold)