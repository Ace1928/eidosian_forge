from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
def _reflection_pad_3d(inputs: Tensor, pad_h: int, pad_w: int, pad_d: int) -> Tensor:
    """Reflective padding of 3d input.

    Args:
        inputs: tensor to pad, should be a 3D tensor of shape ``[N, C, H, W, D]``
        pad_w: amount of padding in the height dimension
        pad_h: amount of padding in the width dimension
        pad_d: amount of padding in the depth dimension

    Returns:
        padded input tensor

    """
    return F.pad(inputs, (pad_h, pad_h, pad_w, pad_w, pad_d, pad_d), mode='reflect')