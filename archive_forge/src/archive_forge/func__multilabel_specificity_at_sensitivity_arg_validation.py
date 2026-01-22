import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_specificity_at_sensitivity_arg_validation(num_labels: int, min_sensitivity: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None) -> None:
    _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
    if not isinstance(min_sensitivity, float) and (not 0 <= min_sensitivity <= 1):
        raise ValueError(f'Expected argument `min_sensitivity` to be an float in the [0,1] range, but got {min_sensitivity}')