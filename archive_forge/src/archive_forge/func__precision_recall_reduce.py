from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _precision_recall_reduce(stat: Literal['precision', 'recall'], tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, average: Optional[Literal['binary', 'micro', 'macro', 'weighted', 'none']], multidim_average: Literal['global', 'samplewise']='global', multilabel: bool=False, top_k: int=1) -> Tensor:
    different_stat = fp if stat == 'precision' else fn
    if average == 'binary':
        return _safe_divide(tp, tp + different_stat)
    if average == 'micro':
        tp = tp.sum(dim=0 if multidim_average == 'global' else 1)
        fn = fn.sum(dim=0 if multidim_average == 'global' else 1)
        different_stat = different_stat.sum(dim=0 if multidim_average == 'global' else 1)
        return _safe_divide(tp, tp + different_stat)
    score = _safe_divide(tp, tp + different_stat)
    return _adjust_weights_safe_divide(score, average, multilabel, tp, fp, fn, top_k=top_k)