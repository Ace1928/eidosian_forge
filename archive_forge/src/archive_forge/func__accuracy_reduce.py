from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _accuracy_reduce(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, average: Optional[Literal['binary', 'micro', 'macro', 'weighted', 'none']], multidim_average: Literal['global', 'samplewise']='global', multilabel: bool=False, top_k: int=1) -> Tensor:
    """Reduce classification statistics into accuracy score.

    Args:
        tp: number of true positives
        fp: number of false positives
        tn: number of true negatives
        fn: number of false negatives
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``binary``: for binary reduction
            - ``micro``: sum score over all classes/labels
            - ``macro``: salculate score for each class/label and average them
            - ``weighted``: calculates score for each class/label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each class/label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.

        multilabel: If input is multilabel or not
        top_k: value for top-k accuracy, else 1

    Returns:
        Accuracy score

    """
    if average == 'binary':
        return _safe_divide(tp + tn, tp + tn + fp + fn)
    if average == 'micro':
        tp = tp.sum(dim=0 if multidim_average == 'global' else 1)
        fn = fn.sum(dim=0 if multidim_average == 'global' else 1)
        if multilabel:
            fp = fp.sum(dim=0 if multidim_average == 'global' else 1)
            tn = tn.sum(dim=0 if multidim_average == 'global' else 1)
            return _safe_divide(tp + tn, tp + tn + fp + fn)
        return _safe_divide(tp, tp + fn)
    score = _safe_divide(tp + tn, tp + tn + fp + fn) if multilabel else _safe_divide(tp, tp + fn)
    return _adjust_weights_safe_divide(score, average, multilabel, tp, fp, fn, top_k)