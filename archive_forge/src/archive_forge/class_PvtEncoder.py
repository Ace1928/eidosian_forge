import collections
import math
from typing import Iterable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_pvt import PvtConfig
class PvtEncoder(nn.Module):

    def __init__(self, config: PvtConfig):
        super().__init__()
        self.config = config
        drop_path_decays = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(PvtPatchEmbeddings(config=config, image_size=config.image_size if i == 0 else self.config.image_size // 2 ** (i + 1), patch_size=config.patch_sizes[i], stride=config.strides[i], num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1], hidden_size=config.hidden_sizes[i], cls_token=i == config.num_encoder_blocks - 1))
        self.patch_embeddings = nn.ModuleList(embeddings)
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(PvtLayer(config=config, hidden_size=config.hidden_sizes[i], num_attention_heads=config.num_attention_heads[i], drop_path=drop_path_decays[cur + j], sequences_reduction_ratio=config.sequence_reduction_ratios[i], mlp_ratio=config.mlp_ratios[i]))
            blocks.append(nn.ModuleList(layers))
        self.block = nn.ModuleList(blocks)
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        batch_size = pixel_values.shape[0]
        num_blocks = len(self.block)
        hidden_states = pixel_values
        for idx, (embedding_layer, block_layer) in enumerate(zip(self.patch_embeddings, self.block)):
            hidden_states, height, width = embedding_layer(hidden_states)
            for block in block_layer:
                layer_outputs = block(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
            if idx != num_blocks - 1:
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)