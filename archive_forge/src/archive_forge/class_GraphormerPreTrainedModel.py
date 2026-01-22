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
class GraphormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GraphormerConfig
    base_model_prefix = 'graphormer'
    main_input_name_nodes = 'input_nodes'
    main_input_name_edges = 'input_edges'

    def normal_(self, data: torch.Tensor):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def init_graphormer_params(self, module: Union[nn.Linear, nn.Embedding, GraphormerMultiheadAttention]):
        """
        Initialize the weights specific to the Graphormer Model.
        """
        if isinstance(module, nn.Linear):
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, GraphormerMultiheadAttention):
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, GraphormerMultiheadAttention, GraphormerGraphEncoder]):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GraphormerMultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, GraphormerGraphEncoder):
            if module.apply_graphormer_init:
                module.apply(self.init_graphormer_params)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)