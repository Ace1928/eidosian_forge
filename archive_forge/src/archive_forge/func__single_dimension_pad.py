from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
def _single_dimension_pad(inputs: Tensor, dim: int, pad: int, outer_pad: int=0) -> Tensor:
    """Apply single-dimension reflection padding to match scipy implementation.

    Args:
        inputs: Input image
        dim: A dimension the image should be padded over
        pad: Number of pads
        outer_pad: Number of outer pads

    Return:
        Image padded over a single dimension

    """
    _max = inputs.shape[dim]
    x = torch.index_select(inputs, dim, torch.arange(pad - 1, -1, -1).to(inputs.device))
    y = torch.index_select(inputs, dim, torch.arange(_max - 1, _max - pad - outer_pad, -1).to(inputs.device))
    return torch.cat((x, inputs, y), dim)