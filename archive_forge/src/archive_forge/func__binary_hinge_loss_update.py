from typing import Optional, Tuple
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _binary_hinge_loss_update(preds: Tensor, target: Tensor, squared: bool) -> Tuple[Tensor, Tensor]:
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