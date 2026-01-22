import inspect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...utils import (
from ..auto import AutoModel
from .configuration_idefics2 import Idefics2Config, Idefics2VisionConfig
class Idefics2PerceiverLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps
        self.input_latents_norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = IDEFICS2_PERCEIVER_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        self.post_attention_layernorm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.mlp = Idefics2MLP(hidden_size=config.text_config.hidden_size, intermediate_size=config.text_config.hidden_size * 4, output_size=config.text_config.hidden_size, hidden_act=config.perceiver_config.hidden_act)

    def forward(self, latents: torch.Tensor, context: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=False, **kwargs) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = latents
        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)
        latents, self_attn_weights, present_key_value = self.self_attn(latents=latents, context=context, attention_mask=attention_mask)
        latents = residual + latents
        residual = latents
        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents
        outputs = (latents,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs