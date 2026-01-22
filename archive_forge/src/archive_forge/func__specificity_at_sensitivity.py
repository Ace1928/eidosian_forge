import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.enums import ClassificationTask
def _specificity_at_sensitivity(specificity: Tensor, sensitivity: Tensor, thresholds: Tensor, min_sensitivity: float) -> Tuple[Tensor, Tensor]:
    indices = sensitivity >= min_sensitivity
    if not indices.any():
        max_spec = torch.tensor(0.0, device=specificity.device, dtype=specificity.dtype)
        best_threshold = torch.tensor(1000000.0, device=thresholds.device, dtype=thresholds.dtype)
    else:
        specificity, sensitivity, thresholds = (specificity[indices], sensitivity[indices], thresholds[indices])
        idx = torch.argmax(specificity)
        max_spec, best_threshold = (specificity[idx], thresholds[idx])
    return (max_spec, best_threshold)