from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.enums import ClassificationTask
def _recall_at_precision(precision: Tensor, recall: Tensor, thresholds: Tensor, min_precision: float) -> Tuple[Tensor, Tensor]:
    max_recall = torch.tensor(0.0, device=recall.device, dtype=recall.dtype)
    best_threshold = torch.tensor(0)
    zipped_len = min((t.shape[0] for t in (recall, precision, thresholds)))
    zipped = torch.vstack((recall[:zipped_len], precision[:zipped_len], thresholds[:zipped_len])).T
    zipped_masked = zipped[zipped[:, 1] >= min_precision]
    if zipped_masked.shape[0] > 0:
        idx = _lexargmax(zipped_masked)[0]
        max_recall, _, best_threshold = zipped_masked[idx]
    if max_recall == 0.0:
        best_threshold = torch.tensor(1000000.0, device=thresholds.device, dtype=thresholds.dtype)
    return (max_recall, best_threshold)