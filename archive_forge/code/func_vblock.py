from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer
def vblock(main_block, hidden_states, attention_mask, position_ids, past_key_value, image_hidden_states, image_attention_mask, cross_attention_gate, output_attentions, use_cache, layer_idx, cross_layer_interval, gated_cross_attn_layers):
    if layer_idx % cross_layer_interval == 0:
        xblock = gated_cross_attn_layers[layer_idx // cross_layer_interval]
        outputs = xblock(hidden_states, attention_mask=attention_mask, image_hidden_states=image_hidden_states, image_attention_mask=image_attention_mask, cross_attention_gate=cross_attention_gate, output_attentions=output_attentions, use_cache=use_cache, past_key_value=None)
        hidden_states = outputs[0]
    layer_outputs = main_block(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
    return layer_outputs