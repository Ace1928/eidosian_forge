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
class SpeechT5SpectrogramLoss(nn.Module):
    """
    Loss computation used by SpeechT5ForTextToSpeech.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.use_guided_attention_loss = config.use_guided_attention_loss
        self.guided_attention_loss_num_heads = config.guided_attention_loss_num_heads
        self.reduction_factor = config.reduction_factor
        self.l1_criterion = L1Loss()
        self.bce_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
        if self.use_guided_attention_loss:
            self.attn_criterion = SpeechT5GuidedMultiheadAttentionLoss(config)

    def forward(self, attention_mask: torch.LongTensor, outputs_before_postnet: torch.FloatTensor, outputs_after_postnet: torch.FloatTensor, logits: torch.FloatTensor, labels: torch.FloatTensor, cross_attentions: Optional[torch.FloatTensor]=None) -> torch.Tensor:
        padding_mask = labels != -100.0
        labels = labels.masked_select(padding_mask)
        outputs_before_postnet = outputs_before_postnet.masked_select(padding_mask)
        outputs_after_postnet = outputs_after_postnet.masked_select(padding_mask)
        l1_loss = self.l1_criterion(outputs_after_postnet, labels) + self.l1_criterion(outputs_before_postnet, labels)
        masks = padding_mask[:, :, 0]
        stop_labels = torch.cat([~masks * 1.0, torch.ones(masks.size(0), 1).to(masks.device)], dim=1)
        stop_labels = stop_labels[:, 1:].masked_select(masks)
        logits = logits.masked_select(masks)
        bce_loss = self.bce_criterion(logits, stop_labels)
        loss = l1_loss + bce_loss
        if self.use_guided_attention_loss:
            attn = torch.cat([x[:, :self.guided_attention_loss_num_heads] for x in cross_attentions], dim=1)
            input_masks = attention_mask == 1
            output_masks = padding_mask[:, :, 0]
            if self.reduction_factor > 1:
                output_masks = output_masks[:, self.reduction_factor - 1::self.reduction_factor]
            attn_loss = self.attn_criterion(attn, input_masks, output_masks)
            loss += attn_loss
        return loss