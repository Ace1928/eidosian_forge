import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_dbrx import DbrxConfig
class DbrxFFN(nn.Module):

    def __init__(self, config: DbrxConfig):
        super().__init__()
        ffn_config = config.ffn_config
        self.router = DbrxRouter(hidden_size=config.d_model, moe_num_experts=ffn_config.moe_num_experts, moe_top_k=ffn_config.moe_top_k, moe_jitter_eps=ffn_config.moe_jitter_eps, moe_normalize_expert_weights=ffn_config.moe_normalize_expert_weights)
        self.experts = DbrxExperts(hidden_size=config.d_model, ffn_hidden_size=ffn_config.ffn_hidden_size, moe_num_experts=ffn_config.moe_num_experts, ffn_act_fn=ffn_config.ffn_act_fn)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, top_weights, top_experts = self.router(x)
        out = self.experts(x, weights, top_weights, top_experts)
        return (out, weights)