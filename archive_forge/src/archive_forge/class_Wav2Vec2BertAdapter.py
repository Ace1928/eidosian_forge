import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wav2vec2_bert import Wav2Vec2BertConfig
class Wav2Vec2BertAdapter(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size, eps=config.layer_norm_eps)
        else:
            self.proj = self.proj_layer_norm = None
        self.layers = nn.ModuleList((Wav2Vec2BertAdapterLayer(config) for _ in range(config.num_adapter_layers)))
        self.layerdrop = config.layerdrop
        self.kernel_size = config.adapter_kernel_size
        self.stride = config.adapter_stride

    def _compute_sub_sample_lengths_from_attention_mask(self, seq_lens):
        if seq_lens is None:
            return seq_lens
        pad = self.kernel_size // 2
        seq_lens = (seq_lens + 2 * pad - self.kernel_size) / self.stride + 1
        return seq_lens.floor()

    def forward(self, hidden_states, attention_mask=None):
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)
        sub_sampled_lengths = None
        if attention_mask is not None:
            sub_sampled_lengths = (attention_mask.size(1) - (1 - attention_mask.int()).sum(1)).to(hidden_states.device)
        for layer in self.layers:
            layerdrop_prob = torch.rand([])
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(sub_sampled_lengths)
            if not self.training or layerdrop_prob > self.layerdrop:
                hidden_states = layer(hidden_states, attention_mask=attention_mask, sub_sampled_lengths=sub_sampled_lengths)
        return hidden_states