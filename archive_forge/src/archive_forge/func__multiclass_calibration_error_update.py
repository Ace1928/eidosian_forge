from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _multiclass_calibration_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.softmax(1)
    confidences, predictions = preds.max(dim=1)
    accuracies = predictions.eq(target)
    return (confidences.float(), accuracies.float())