import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d, pad
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _hp_2d_laplacian(input_img: Tensor, kernel: Tensor) -> Tensor:
    """Applies 2-D Laplace filter to the input tensor with the given high pass filter."""
    return _signal_convolve_2d(input_img, kernel) * 2.0