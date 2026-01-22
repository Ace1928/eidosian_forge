import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None):
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]
    seq_length = input_shape[1]
    if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    if inputs_embeds is None:
        inputs_embeds = self.embeddings.word_embeddings(input_ids)
    position_embeddings = self.embeddings.position_embeddings(position_ids)
    spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
    token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
    embeddings = inputs_embeds + position_embeddings + spatial_position_embeddings + token_type_embeddings
    embeddings = self.embeddings.LayerNorm(embeddings)
    embeddings = self.embeddings.dropout(embeddings)
    return embeddings