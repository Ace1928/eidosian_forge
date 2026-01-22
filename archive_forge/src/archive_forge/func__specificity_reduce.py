from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _specificity_reduce(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, average: Optional[Literal['binary', 'micro', 'macro', 'weighted', 'none']], multidim_average: Literal['global', 'samplewise']='global', multilabel: bool=False) -> Tensor:
    if average == 'binary':
        return _safe_divide(tn, tn + fp)
    if average == 'micro':
        tn = tn.sum(dim=0 if multidim_average == 'global' else 1)
        fp = fp.sum(dim=0 if multidim_average == 'global' else 1)
        return _safe_divide(tn, tn + fp)
    specificity_score = _safe_divide(tn, tn + fp)
    return _adjust_weights_safe_divide(specificity_score, average, multilabel, tp, fp, fn)