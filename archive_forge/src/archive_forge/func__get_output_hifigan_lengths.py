import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
def _get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
    """
        Computes the output length of the hifigan convolutional layers
        """

    def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
        return torch.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode='floor') + 1

    def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
        return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1
    input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
    for i, (upsample_rate, kernel_size) in enumerate(zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)):
        input_lengths = _transpose_conv_out_length(input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2)
    for i in range(len(self.config.upsample_rates)):
        for kernel_size, dilation in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
            for dil in dilation:
                input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil)
            for dil in dilation:
                input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)
    input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
    return input_lengths