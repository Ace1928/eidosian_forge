from typing import Optional, Tuple
import torch
def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed
        """
    batch_size, num_channels, num_frames = input.shape
    is_odd = self.enc_kernel_size % 2
    num_strides = (num_frames - is_odd) // self.enc_stride
    num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
    if num_remainings == 0:
        return (input, 0)
    num_paddings = self.enc_stride - num_remainings
    pad = torch.zeros(batch_size, num_channels, num_paddings, dtype=input.dtype, device=input.device)
    return (torch.cat([input, pad], 2), num_paddings)