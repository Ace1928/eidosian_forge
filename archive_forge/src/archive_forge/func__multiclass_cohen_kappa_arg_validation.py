from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _multiclass_cohen_kappa_arg_validation(num_classes: int, ignore_index: Optional[int]=None, weights: Optional[Literal['linear', 'quadratic', 'none']]=None) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``ignore_index`` has to be None or int
    - ``weights`` has to be "linear" | "quadratic" | "none" | None

    """
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize=None)
    allowed_weights = ('linear', 'quadratic', 'none', None)
    if weights not in allowed_weights:
        raise ValueError(f'Expected argument `weight` to be one of {allowed_weights}, but got {weights}.')