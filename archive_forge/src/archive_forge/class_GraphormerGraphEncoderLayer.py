import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
class GraphormerGraphEncoderLayer(nn.Module):

    def __init__(self, config: GraphormerConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_attention_heads = config.num_attention_heads
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.pre_layernorm = config.pre_layernorm
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.activation_dropout_module = torch.nn.Dropout(p=config.activation_dropout, inplace=False)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.self_attn = GraphormerMultiheadAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = self.build_fc(self.embedding_dim, config.ffn_embedding_dim, q_noise=config.q_noise, qn_block_size=config.qn_block_size)
        self.fc2 = self.build_fc(config.ffn_embedding_dim, self.embedding_dim, q_noise=config.q_noise, qn_block_size=config.qn_block_size)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def build_fc(self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int) -> Union[nn.Module, nn.Linear, nn.Embedding, nn.Conv2d]:
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(self, input_nodes: torch.Tensor, self_attn_bias: Optional[torch.Tensor]=None, self_attn_mask: Optional[torch.Tensor]=None, self_attn_padding_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)
        input_nodes, attn = self.self_attn(query=input_nodes, key=input_nodes, value=input_nodes, attn_bias=self_attn_bias, key_padding_mask=self_attn_padding_mask, need_weights=False, attn_mask=self_attn_mask)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        return (input_nodes, attn)