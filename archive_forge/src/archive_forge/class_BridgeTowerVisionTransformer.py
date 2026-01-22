import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
class BridgeTowerVisionTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embeddings = BridgeTowerVisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.transformer = BridgeTowerTransformer(config)
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.share_layernorm = config.share_layernorm
        if not config.share_layernorm:
            self.ln_separate = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(config.num_hidden_layers)])

    def forward(self, pixel_values: torch.Tensor, attention_mask):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.ln_pre(hidden_states)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.transformer(hidden_states, attention_mask)
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        if self.share_layernorm:
            hidden_states = self.ln_post(hidden_states)
        else:
            hidden_states_stack = []
            for hidden_states, ln in zip(hidden_states, self.ln_separate):
                hidden_states = ln(hidden_states)
                hidden_states_stack.append(hidden_states)
            hidden_states = torch.stack(hidden_states_stack, dim=0)
        return hidden_states

    def forward_pre(self, pixel_values: torch.Tensor):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.ln_pre(hidden_states)
        hidden_states = hidden_states.permute(1, 0, 2)
        return hidden_states

    def forward_post(self, hidden_state: torch.Tensor):
        visual_output_post = hidden_state.permute(1, 0, 2)
        visual_output_post = self.ln_post(visual_output_post)
        return visual_output_post