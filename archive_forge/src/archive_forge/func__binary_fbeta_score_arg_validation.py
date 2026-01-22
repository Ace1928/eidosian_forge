from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _binary_fbeta_score_arg_validation(beta: float, threshold: float=0.5, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None) -> None:
    if not (isinstance(beta, float) and beta > 0):
        raise ValueError(f'Expected argument `beta` to be a float larger than 0, but got {beta}.')
    _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)