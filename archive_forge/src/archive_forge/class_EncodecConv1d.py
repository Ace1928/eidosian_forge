import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
class EncodecConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""

    def __init__(self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, dilation: int=1):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode
        self.norm_type = config.norm_type
        if self.norm_type not in ['weight_norm', 'time_group_norm']:
            raise ValueError(f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}')
        if stride > 1 and dilation > 1:
            logger.warning(f'EncodecConv1d has been initialized with stride > 1 and dilation > 1 (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        if self.norm_type == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif self.norm_type == 'time_group_norm':
            self.norm = nn.GroupNorm(1, out_channels)

    @staticmethod
    def _get_extra_padding_for_conv1d(hidden_states: torch.Tensor, kernel_size: int, stride: int, padding_total: int=0) -> int:
        """See `pad_for_conv1d`."""
        length = hidden_states.shape[-1]
        n_frames = (length - kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        return ideal_length - length

    @staticmethod
    def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str='zero', value: float=0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if not mode == 'reflect':
            return nn.functional.pad(hidden_states, paddings, mode, value)
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def forward(self, hidden_states):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = kernel_size - stride
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states, kernel_size, stride, padding_total)
        if self.causal:
            hidden_states = self._pad1d(hidden_states, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            hidden_states = self._pad1d(hidden_states, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        hidden_states = self.conv(hidden_states)
        if self.norm_type == 'time_group_norm':
            hidden_states = self.norm(hidden_states)
        return hidden_states