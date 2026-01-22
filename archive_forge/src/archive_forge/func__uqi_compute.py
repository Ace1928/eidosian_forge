from typing import Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _uqi_compute(preds: Tensor, target: Tensor, kernel_size: Sequence[int]=(11, 11), sigma: Sequence[float]=(1.5, 1.5), reduction: Optional[Literal['elementwise_mean', 'sum', 'none']]='elementwise_mean') -> Tensor:
    """Compute Universal Image Quality Index.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> preds, target = _uqi_update(preds, target)
        >>> _uqi_compute(preds, target)
        tensor(0.9216)

    """
    if len(kernel_size) != 2 or len(sigma) != 2:
        raise ValueError(f'Expected `kernel_size` and `sigma` to have the length of two. Got kernel_size: {len(kernel_size)} and sigma: {len(sigma)}.')
    if any((x % 2 == 0 or x <= 0 for x in kernel_size)):
        raise ValueError(f'Expected `kernel_size` to have odd positive number. Got {kernel_size}.')
    if any((y <= 0 for y in sigma)):
        raise ValueError(f'Expected `sigma` to have positive number. Got {sigma}.')
    device = preds.device
    channel = preds.size(1)
    dtype = preds.dtype
    kernel = _gaussian_kernel_2d(channel, kernel_size, sigma, dtype, device)
    pad_h = (kernel_size[0] - 1) // 2
    pad_w = (kernel_size[1] - 1) // 2
    preds = nn.functional.pad(preds, (pad_h, pad_h, pad_w, pad_w), mode='reflect')
    target = nn.functional.pad(target, (pad_h, pad_h, pad_w, pad_w), mode='reflect')
    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))
    outputs = nn.functional.conv2d(input_list, kernel, groups=channel)
    output_list = outputs.split(preds.shape[0])
    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]
    sigma_pred_sq = torch.clamp(output_list[2] - mu_pred_sq, min=0.0)
    sigma_target_sq = torch.clamp(output_list[3] - mu_target_sq, min=0.0)
    sigma_pred_target = output_list[4] - mu_pred_target
    upper = 2 * sigma_pred_target
    lower = sigma_pred_sq + sigma_target_sq
    eps = torch.finfo(sigma_pred_sq.dtype).eps
    uqi_idx = 2 * mu_pred_target * upper / ((mu_pred_sq + mu_target_sq) * lower + eps)
    uqi_idx = uqi_idx[..., pad_h:-pad_h, pad_w:-pad_w]
    return reduce(uqi_idx, reduction)