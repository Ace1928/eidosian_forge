import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_whisper import WhisperConfig
from .generation_whisper import WhisperGenerationMixin
def _mask_input_features(self, input_features: torch.FloatTensor, attention_mask: Optional[torch.LongTensor]=None):
    """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
    if not getattr(self.config, 'apply_spec_augment', True):
        return input_features
    batch_size, hidden_size, sequence_length = input_features.size()
    if self.config.mask_time_prob > 0 and self.training:
        mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=self.config.mask_time_prob, mask_length=self.config.mask_time_length, attention_mask=attention_mask, min_masks=self.config.mask_time_min_masks)
        mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
        mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
        input_features[mask_time_indices] = 0
    if self.config.mask_feature_prob > 0 and self.training:
        mask_feature_indices = _compute_mask_indices((batch_size, hidden_size), mask_prob=self.config.mask_feature_prob, mask_length=self.config.mask_feature_length, min_masks=self.config.mask_feature_min_masks)
        mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
        input_features[mask_feature_indices] = 0
    return input_features