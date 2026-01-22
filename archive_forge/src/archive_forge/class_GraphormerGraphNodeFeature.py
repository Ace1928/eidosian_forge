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
class GraphormerGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_atoms = config.num_atoms
        self.atom_encoder = nn.Embedding(config.num_atoms + 1, config.hidden_size, padding_idx=config.pad_token_id)
        self.in_degree_encoder = nn.Embedding(config.num_in_degree, config.hidden_size, padding_idx=config.pad_token_id)
        self.out_degree_encoder = nn.Embedding(config.num_out_degree, config.hidden_size, padding_idx=config.pad_token_id)
        self.graph_token = nn.Embedding(1, config.hidden_size)

    def forward(self, input_nodes: torch.LongTensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor) -> torch.Tensor:
        n_graph, n_node = input_nodes.size()[:2]
        node_feature = self.atom_encoder(input_nodes).sum(dim=-2) + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature