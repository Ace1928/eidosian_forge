from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTask
def _matthews_corrcoef_reduce(confmat: Tensor) -> Tensor:
    """Reduce an un-normalized confusion matrix of shape (n_classes, n_classes) into the matthews corrcoef score.

    See: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7 for more info.

    """
    confmat = confmat.sum(0) if confmat.ndim == 3 else confmat
    if confmat.numel() == 4:
        tn, fp, fn, tp = confmat.reshape(-1)
        if tp + tn != 0 and fp + fn == 0:
            return torch.tensor(1.0, dtype=confmat.dtype, device=confmat.device)
        if tp + tn == 0 and fp + fn != 0:
            return torch.tensor(-1.0, dtype=confmat.dtype, device=confmat.device)
    tk = confmat.sum(dim=-1).float()
    pk = confmat.sum(dim=-2).float()
    c = torch.trace(confmat).float()
    s = confmat.sum().float()
    cov_ytyp = c * s - sum(tk * pk)
    cov_ypyp = s ** 2 - sum(pk * pk)
    cov_ytyt = s ** 2 - sum(tk * tk)
    numerator = cov_ytyp
    denom = cov_ypyp * cov_ytyt
    if denom == 0 and confmat.numel() == 4:
        if tp == 0 or tn == 0:
            a = tp + tn
        if fp == 0 or fn == 0:
            b = fp + fn
        eps = torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float32, device=confmat.device)
        numerator = torch.sqrt(eps) * (a - b)
        denom = (tp + fp + eps) * (tp + fn + eps) * (tn + fp + eps) * (tn + fn + eps)
    elif denom == 0:
        return torch.tensor(0, dtype=confmat.dtype, device=confmat.device)
    return numerator / torch.sqrt(denom)