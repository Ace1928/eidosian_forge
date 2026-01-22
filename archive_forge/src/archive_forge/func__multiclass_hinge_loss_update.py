from typing import Optional, Tuple
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _multiclass_hinge_loss_update(preds: Tensor, target: Tensor, squared: bool, multiclass_mode: Literal['crammer-singer', 'one-vs-all']='crammer-singer') -> Tuple[Tensor, Tensor]:
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.softmax(1)
    target = to_onehot(target, max(2, preds.shape[1])).bool()
    if multiclass_mode == 'crammer-singer':
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    else:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]
    measures = 1 - margin
    measures = torch.clamp(measures, 0)
    if squared:
        measures = measures.pow(2)
    total = tensor(target.shape[0], device=target.device)
    return (measures.sum(dim=0), total)