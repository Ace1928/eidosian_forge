from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_roc_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_labels: int, thresholds: Optional[Tensor], ignore_index: Optional[int]=None) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        tns = state[:, :, 0, 0]
        tpr = _safe_divide(tps, tps + fns).flip(0).T
        fpr = _safe_divide(fps, fps + tns).flip(0).T
        thres = thresholds.flip(0)
    else:
        fpr, tpr, thres = ([], [], [])
        for i in range(num_labels):
            preds = state[0][:, i]
            target = state[1][:, i]
            if ignore_index is not None:
                idx = target == ignore_index
                preds = preds[~idx]
                target = target[~idx]
            res = _binary_roc_compute((preds, target), thresholds=None, pos_label=1)
            fpr.append(res[0])
            tpr.append(res[1])
            thres.append(res[2])
    return (fpr, tpr, thres)