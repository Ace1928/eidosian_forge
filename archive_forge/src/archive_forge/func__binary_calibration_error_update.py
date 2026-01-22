from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _binary_calibration_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    confidences, accuracies = (preds, target)
    return (confidences, accuracies)