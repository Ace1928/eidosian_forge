import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_poolformer import PoolFormerConfig
class PoolFormerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(PoolFormerEmbeddings(patch_size=config.patch_sizes[i], stride=config.strides[i], padding=config.padding[i], num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1], hidden_size=config.hidden_sizes[i]))
        self.patch_embeddings = nn.ModuleList(embeddings)
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(PoolFormerLayer(config, num_channels=config.hidden_sizes[i], pool_size=config.pool_size, hidden_size=config.hidden_sizes[i], intermediate_size=int(config.hidden_sizes[i] * config.mlp_ratio), drop_path=dpr[cur + j]))
            blocks.append(nn.ModuleList(layers))
        self.block = nn.ModuleList(blocks)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        hidden_states = pixel_values
        for idx, layers in enumerate(zip(self.patch_embeddings, self.block)):
            embedding_layer, block_layer = layers
            hidden_states = embedding_layer(hidden_states)
            for _, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states)
                hidden_states = layer_outputs[0]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)