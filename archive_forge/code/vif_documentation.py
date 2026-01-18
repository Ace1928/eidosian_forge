import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torchmetrics.utilities.distributed import reduce
Compute Pixel Based Visual Information Fidelity (VIF_).

    Args:
        preds: predicted images of shape ``(N,C,H,W)``. ``(H, W)`` has to be at least ``(41, 41)``.
        target: ground truth images of shape ``(N,C,H,W)``. ``(H, W)`` has to be at least ``(41, 41)``
        sigma_n_sq: variance of the visual noise

    Return:
        Tensor with vif-p score

    Raises:
        ValueError:
            If ``data_range`` is neither a ``tuple`` nor a ``float``

    