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
class GraphormerGraphEncoder(nn.Module):

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_graphormer_init = config.apply_graphormer_init
        self.traceable = config.traceable
        self.graph_node_feature = GraphormerGraphNodeFeature(config)
        self.graph_attn_bias = GraphormerGraphAttnBias(config)
        self.embed_scale = config.embed_scale
        if config.q_noise > 0:
            self.quant_noise = quant_noise(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), config.q_noise, config.qn_block_size)
        else:
            self.quant_noise = None
        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None
        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([GraphormerGraphEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        if config.freeze_embeddings:
            raise NotImplementedError('Freezing embeddings is not implemented yet.')
        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, perturb=None, last_state_only: bool=False, token_embeddings: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None) -> Tuple[Union[torch.Tensor, List[torch.LongTensor]], torch.Tensor]:
        data_x = input_nodes
        n_graph, n_node = data_x.size()[:2]
        padding_mask = data_x[:, :, 0].eq(0)
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type)
        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            input_nodes = self.graph_node_feature(input_nodes, in_degree, out_degree)
        if perturb is not None:
            input_nodes[:, 1:, :] += perturb
        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale
        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)
        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = input_nodes.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(input_nodes)
        for layer in self.layers:
            input_nodes, _ = layer(input_nodes, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)
            if not last_state_only:
                inner_states.append(input_nodes)
        graph_rep = input_nodes[0, :, :]
        if last_state_only:
            inner_states = [input_nodes]
        if self.traceable:
            return (torch.stack(inner_states), graph_rep)
        else:
            return (inner_states, graph_rep)