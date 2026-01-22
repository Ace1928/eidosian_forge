import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ....modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_open_llama import OpenLlamaConfig
def _init_rope(self):
    if self.config.rope_scaling is None:
        self.rotary_emb = OpenLlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
    else:
        scaling_type = self.config.rope_scaling['type']
        scaling_factor = self.config.rope_scaling['factor']
        if scaling_type == 'linear':
            self.rotary_emb = OpenLlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor, base=self.rope_theta)
        elif scaling_type == 'dynamic':
            self.rotary_emb = OpenLlamaDynamicNTKScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor, base=self.rope_theta)
        else:
            raise ValueError(f'Unknown RoPE scaling type {scaling_type}')