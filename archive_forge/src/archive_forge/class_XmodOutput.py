import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xmod import XmodConfig
class XmodOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_before_adapter = config.ln_before_adapter
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.adapter_layer_norm:
            self.adapter_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.adapter_layer_norm = None
        self.adapter_reuse_layer_norm = config.adapter_reuse_layer_norm
        self.adapter_modules = nn.ModuleDict({})
        for language in config.languages:
            self.adapter_modules[str(language)] = XmodAdapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.lang_adapter(lang_ids, hidden_states)
        return hidden_states

    def lang_adapter(self, lang_ids: torch.Tensor, hidden_states: torch.Tensor):
        lang_ids, lang_lengths = torch.unique_consecutive(lang_ids, return_counts=True)
        if not self.ln_before_adapter:
            residual = hidden_states
        if self.adapter_layer_norm is not None:
            hidden_states = self.adapter_layer_norm(hidden_states)
        elif self.adapter_reuse_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        if self.ln_before_adapter:
            residual = hidden_states
        split_hidden_states = torch.split(hidden_states, lang_lengths.tolist(), 0)
        lang_wise_outputs = []
        for i, (lang_id, split_hidden_state) in enumerate(zip(lang_ids, split_hidden_states)):
            lang = list(self.adapter_modules.keys())[int(lang_id.item())]
            lang_wise_outputs.append(self.adapter_modules[lang](split_hidden_state))
        hidden_states = torch.cat(lang_wise_outputs, 0)
        hidden_states = self.dropout(hidden_states)
        hidden_states += residual
        return hidden_states