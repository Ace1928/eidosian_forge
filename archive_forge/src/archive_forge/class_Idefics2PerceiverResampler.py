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
class Idefics2PerceiverResampler(nn.Module):

    def __init__(self, config) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim]. The Resampler acts as a form of learned pooling and
        is derived from [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206).
        """
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.hidden_act = config.perceiver_config.hidden_act
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps
        self.latents = nn.Parameter(torch.ones(self.n_latents, self.hidden_size))
        self.layers = nn.ModuleList([Idefics2PerceiverLayer(config, idx) for idx in range(self.depth)])
        self.norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == 'flash_attention_2'

    def forward(self, context: torch.Tensor, attention_mask) -> torch.Tensor:
        latents = self.latents.unsqueeze(0).expand((context.shape[0], *self.latents.size()))
        latent_attention_mask = torch.ones((attention_mask.size(0), latents.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latent_attention_mask], dim=-1)
        attention_mask = _prepare_4d_attention_mask(attention_mask, latents.dtype, tgt_len=self.n_latents) if not self._use_flash_attention_2 else attention_mask
        compressed_context = latents
        for perceiver_layer in self.layers:
            layer_outputs = perceiver_layer(compressed_context, context, attention_mask=attention_mask, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False)
            compressed_context = layer_outputs[0]
        compressed_context = self.norm(compressed_context)
        return compressed_context