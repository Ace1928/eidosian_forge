from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d

        Args:
            x (Tensor): input of shape ``(batch_size, channels, time_length)``.
        Returns:
            Tensor of the same shape as input.
        