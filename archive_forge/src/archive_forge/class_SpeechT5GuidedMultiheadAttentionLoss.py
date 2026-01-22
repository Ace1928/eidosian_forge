import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
class SpeechT5GuidedMultiheadAttentionLoss(nn.Module):
    """
    Guided attention loss from the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional
    Networks with Guided Attention](https://arxiv.org/abs/1710.08969), adapted for multi-head attention.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.sigma = config.guided_attention_loss_sigma
        self.scale = config.guided_attention_loss_scale

    def forward(self, attentions: torch.FloatTensor, input_masks: torch.BoolTensor, output_masks: torch.BoolTensor) -> torch.Tensor:
        """
        Compute the attention loss.

        Args:
            attentions (`torch.FloatTensor` of shape `(batch_size, layers * heads, output_sequence_length, input_sequence_length)`):
                Batch of multi-head attention weights
            input_masks (`torch.BoolTensor` of shape `(batch_size, input_sequence_length)`):
                Input attention mask as booleans.
            output_masks (`torch.BoolTensor` of shape `(batch_size, output_sequence_length)`):
                Target attention mask as booleans.

        Returns:
            `torch.Tensor` with the loss value
        """
        guided_attn_masks = self._make_guided_attention_masks(input_masks, output_masks, attentions.device)
        masks = output_masks.unsqueeze(-1) & input_masks.unsqueeze(-2)
        masks = masks.to(attentions.device).unsqueeze(1)
        losses = guided_attn_masks * attentions
        loss = torch.mean(losses.masked_select(masks))
        return self.scale * loss

    def _make_guided_attention_masks(self, input_masks, output_masks, device):
        input_lengths = input_masks.sum(-1)
        output_lengths = output_masks.sum(-1)
        guided_attn_masks = torch.zeros((len(input_masks), output_masks.shape[1], input_masks.shape[1]), device=device)
        for idx, (ilen, olen) in enumerate(zip(input_lengths, output_lengths)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma, device)
        return guided_attn_masks.unsqueeze(1)

    @staticmethod
    def _make_guided_attention_mask(input_length, output_length, sigma, device):
        grid_y, grid_x = torch.meshgrid(torch.arange(input_length, device=device), torch.arange(output_length, device=device), indexing='xy')
        grid_x = grid_x.float() / output_length
        grid_y = grid_y.float() / input_length
        return 1.0 - torch.exp(-(grid_y - grid_x) ** 2 / (2 * sigma ** 2))