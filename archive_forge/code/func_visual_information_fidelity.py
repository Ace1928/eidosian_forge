import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torchmetrics.utilities.distributed import reduce
def visual_information_fidelity(preds: Tensor, target: Tensor, sigma_n_sq: float=2.0) -> Tensor:
    """Compute Pixel Based Visual Information Fidelity (VIF_).

    Args:
        preds: predicted images of shape ``(N,C,H,W)``. ``(H, W)`` has to be at least ``(41, 41)``.
        target: ground truth images of shape ``(N,C,H,W)``. ``(H, W)`` has to be at least ``(41, 41)``
        sigma_n_sq: variance of the visual noise

    Return:
        Tensor with vif-p score

    Raises:
        ValueError:
            If ``data_range`` is neither a ``tuple`` nor a ``float``

    """
    if preds.size(-1) < 41 or preds.size(-2) < 41:
        raise ValueError(f'Invalid size of preds. Expected at least 41x41, but got {preds.size(-1)}x{preds.size(-2)}!')
    if target.size(-1) < 41 or target.size(-2) < 41:
        raise ValueError(f'Invalid size of target. Expected at least 41x41, but got {target.size(-1)}x{target.size(-2)}!')
    per_channel = [_vif_per_channel(preds[:, i, :, :], target[:, i, :, :], sigma_n_sq) for i in range(preds.size(1))]
    return reduce(torch.cat(per_channel), 'elementwise_mean')