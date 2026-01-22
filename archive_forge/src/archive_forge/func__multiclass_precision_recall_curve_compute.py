from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multiclass_precision_recall_curve_compute(state: Union[Tensor, Tuple[Tensor, Tensor]], num_classes: int, thresholds: Optional[Tensor], average: Optional[Literal['micro', 'macro']]=None) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve in an iterative way.

    """
    if average == 'micro':
        return _binary_precision_recall_curve_compute(state, thresholds)
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        precision = _safe_divide(tps, tps + fps)
        recall = _safe_divide(tps, tps + fns)
        precision = torch.cat([precision, torch.ones(1, num_classes, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([recall, torch.zeros(1, num_classes, dtype=recall.dtype, device=recall.device)])
        precision = precision.T
        recall = recall.T
        thres = thresholds
        tensor_state = True
    else:
        precision_list, recall_list, thres_list = ([], [], [])
        for i in range(num_classes):
            res = _binary_precision_recall_curve_compute((state[0][:, i], state[1]), thresholds=None, pos_label=i)
            precision_list.append(res[0])
            recall_list.append(res[1])
            thres_list.append(res[2])
        tensor_state = False
    if average == 'macro':
        thres = thres.repeat(num_classes) if tensor_state else torch.cat(thres_list, 0)
        thres = thres.sort().values
        mean_precision = precision.flatten() if tensor_state else torch.cat(precision_list, 0)
        mean_precision = mean_precision.sort().values
        mean_recall = torch.zeros_like(mean_precision)
        for i in range(num_classes):
            mean_recall += interp(mean_precision, precision[i] if tensor_state else precision_list[i], recall[i] if tensor_state else recall_list[i])
        mean_recall /= num_classes
        return (mean_precision, mean_recall, thres)
    if tensor_state:
        return (precision, recall, thres)
    return (precision_list, recall_list, thres_list)