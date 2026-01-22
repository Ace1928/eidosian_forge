import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.enums import ClassificationTask
def _binary_specificity_at_sensitivity_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], thresholds: Optional[Tensor], min_sensitivity: float, pos_label: int=1) -> Tuple[Tensor, Tensor]:
    fpr, sensitivity, thresholds = _binary_roc_compute(state, thresholds, pos_label)
    specificity = _convert_fpr_to_specificity(fpr)
    return _specificity_at_sensitivity(specificity, sensitivity, thresholds, min_sensitivity)