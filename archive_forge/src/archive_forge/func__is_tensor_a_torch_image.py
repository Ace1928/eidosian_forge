import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2