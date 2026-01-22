import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once

        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
                offsets to be applied for each position in the convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
                masks to be applied for each position in the convolution kernel.
        