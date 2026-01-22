import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_pvt_v2 import PvtV2Config
class PvtV2Encoder(nn.Module):

    def __init__(self, config: PvtV2Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([PvtV2EncoderLayer(config, i) for i in range(config.num_encoder_blocks)])

    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        batch_size = pixel_values.shape[0]
        hidden_states = pixel_values
        for idx, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_output = self._gradient_checkpointing_func(layer.__call__, hidden_states, output_attentions)
            else:
                layer_output = layer(hidden_states, output_attentions)
            outputs, height, width = layer_output
            hidden_states = outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)