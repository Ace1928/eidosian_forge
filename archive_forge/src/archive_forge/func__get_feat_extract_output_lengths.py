import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
from ....utils import logging
from .configuration_mctct import MCTCTConfig
def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
    """
        Computes the output length of the convolutional layers
        """
    dilation = 1
    for _, kernel_sz, stride in zip(range(self.config.num_conv_layers), self.config.conv_kernel, self.config.conv_stride):
        padding = kernel_sz // 2
        input_lengths = input_lengths + 2 * padding - dilation * (kernel_sz - 1) - 1
        input_lengths = torch.div(input_lengths, stride, rounding_mode='trunc') + 1
    return input_lengths