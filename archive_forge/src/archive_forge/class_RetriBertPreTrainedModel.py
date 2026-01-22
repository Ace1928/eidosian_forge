import math
from typing import Optional
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, logging
from ...bert.modeling_bert import BertModel
from .configuration_retribert import RetriBertConfig
class RetriBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RetriBertConfig
    load_tf_weights = None
    base_model_prefix = 'retribert'

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)