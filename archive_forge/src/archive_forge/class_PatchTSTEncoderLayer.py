import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
class PatchTSTEncoderLayer(nn.Module):
    """
    PatchTST encoder layer
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.channel_attention = config.channel_attention
        self.self_attn = PatchTSTAttention(embed_dim=config.d_model, num_heads=config.num_attention_heads, dropout=config.attention_dropout)
        self.dropout_path1 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        if config.norm_type == 'batchnorm':
            self.norm_sublayer1 = PatchTSTBatchNorm(config)
        elif config.norm_type == 'layernorm':
            self.norm_sublayer1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        else:
            raise ValueError(f'{config.norm_type} is not a supported norm layer type.')
        if self.channel_attention:
            self.dropout_path2 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
            if config.norm_type == 'batchnorm':
                self.norm_sublayer2 = PatchTSTBatchNorm(config)
            elif config.norm_type == 'layernorm':
                self.norm_sublayer2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            else:
                raise ValueError(f'{config.norm_type} is not a supported norm layer type.')
        self.ff = nn.Sequential(nn.Linear(config.d_model, config.ffn_dim, bias=config.bias), ACT2CLS[config.activation_function](), nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(), nn.Linear(config.ffn_dim, config.d_model, bias=config.bias))
        self.dropout_path3 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        if config.norm_type == 'batchnorm':
            self.norm_sublayer3 = PatchTSTBatchNorm(config)
        elif config.norm_type == 'layernorm':
            self.norm_sublayer3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        else:
            raise ValueError(f'{config.norm_type} is not a supported norm layer type.')
        self.pre_norm = config.pre_norm

    def forward(self, hidden_state: torch.Tensor, output_attentions: Optional[bool]=None):
        """
        Parameters:
            hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`, *required*):
                Past values of the time series
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
        Return:
            `torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`

        """
        batch_size, num_input_channels, sequence_length, d_model = hidden_state.shape
        hidden_state = hidden_state.view(batch_size * num_input_channels, sequence_length, d_model)
        if self.pre_norm:
            attn_output, attn_weights, _ = self.self_attn(hidden_states=self.norm_sublayer1(hidden_state), output_attentions=output_attentions)
            hidden_state = hidden_state + self.dropout_path1(attn_output)
        else:
            attn_output, attn_weights, _ = self.self_attn(hidden_states=hidden_state, output_attentions=output_attentions)
            hidden_state = self.norm_sublayer1(hidden_state + self.dropout_path1(attn_output))
        hidden_state = hidden_state.reshape(batch_size, num_input_channels, sequence_length, d_model)
        if self.channel_attention:
            hidden_state = hidden_state.transpose(2, 1).contiguous()
            hidden_state = hidden_state.view(batch_size * sequence_length, num_input_channels, d_model)
            if self.pre_norm:
                attn_output, channel_attn_weights, _ = self.self_attn(hidden_states=self.norm_sublayer2(hidden_state), output_attentions=output_attentions)
                hidden_state = hidden_state + self.dropout_path2(attn_output)
            else:
                attn_output, channel_attn_weights, _ = self.self_attn(hidden_states=hidden_state, output_attentions=output_attentions)
                hidden_state = self.norm_sublayer2(hidden_state + self.dropout_path2(attn_output))
            hidden_state = hidden_state.reshape(batch_size, sequence_length, num_input_channels, d_model)
            hidden_state = hidden_state.transpose(1, 2).contiguous()
        hidden_state = hidden_state.view(batch_size * num_input_channels, sequence_length, d_model)
        if self.pre_norm:
            hidden_state = hidden_state + self.dropout_path3(self.ff(self.norm_sublayer3(hidden_state)))
        else:
            hidden_state = self.norm_sublayer3(hidden_state + self.dropout_path3(self.ff(hidden_state)))
        hidden_state = hidden_state.reshape(batch_size, num_input_channels, sequence_length, d_model)
        outputs = (hidden_state,)
        if output_attentions:
            outputs += (attn_weights, channel_attn_weights) if self.channel_attention else (attn_weights,)
        return outputs