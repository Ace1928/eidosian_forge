import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d, pad
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
Compute Spatial Correlation Coefficient (SCC_).

    Args:
        preds: predicted images of shape ``(N,C,H,W)`` or ``(N,H,W)``.
        target: ground truth images of shape ``(N,C,H,W)`` or ``(N,H,W)``.
        hp_filter: High-pass filter tensor. default: tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        window_size: Local window size integer. default: 8,
        reduction: Reduction method for output tensor. If ``None`` or ``"none"``,
                   returns a tensor with the per sample results. default: ``"mean"``.

    Return:
        Tensor with scc score

    Example:
        >>> import torch
        >>> from torchmetrics.functional.image import spatial_correlation_coefficient as scc
        >>> _ = torch.manual_seed(42)
        >>> x = torch.randn(5, 3, 16, 16)
        >>> scc(x, x)
        tensor(1.)
        >>> x = torch.randn(5, 16, 16)
        >>> scc(x, x)
        tensor(1.)
        >>> x = torch.randn(5, 3, 16, 16)
        >>> y = torch.randn(5, 3, 16, 16)
        >>> scc(x, y, reduction="none")
        tensor([0.0223, 0.0256, 0.0616, 0.0159, 0.0170])

    