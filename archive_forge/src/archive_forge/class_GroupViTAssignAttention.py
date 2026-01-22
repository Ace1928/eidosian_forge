import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class GroupViTAssignAttention(nn.Module):

    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()
        self.scale = config.hidden_size ** (-0.5)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.assign_eps = config.assign_eps

    def get_attn(self, attn, gumbel=True, hard=True):
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        elif hard:
            attn = hard_softmax(attn, dim=-2)
        else:
            attn = nn.functional.softmax(attn, dim=-2)
        return attn

    def forward(self, query, key):
        value = key
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        raw_attn = query @ key.transpose(-2, -1) * self.scale
        attn = self.get_attn(raw_attn)
        soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        out = attn @ value
        out = self.proj(out)
        return (out, soft_attn)