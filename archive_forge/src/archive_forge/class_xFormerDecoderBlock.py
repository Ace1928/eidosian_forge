import logging
from dataclasses import asdict
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers._deprecation_warning import deprecated_function
from xformers.components import (
from xformers.components.attention import AttentionMask
from xformers.components.feedforward import build_feedforward
from xformers.components.positional_embedding import build_positional_embedding
from xformers.components.residual import get_deepnorm_coefficients
from xformers.components.simplicial_embedding import SimplicialEmbedding
from xformers.factory.block_configs import (
class xFormerDecoderBlock(torch.nn.Module):
    """A vanilla Transformer Decoder block

    ... note: this implementation is not (yet ?) reversible"""

    def __init__(self, config: xFormerDecoderConfig, **kwargs):
        super().__init__()
        deprecated_function(self)
        if config.position_encoding_config is not None and config.layer_position.is_first():
            self.pose_encoding = build_positional_embedding(config.position_encoding_config)
            pos_encoding_dim = config.position_encoding_config.dim_model
            mha_dim = config.multi_head_config_masked['dim_model']
            if pos_encoding_dim != mha_dim:
                logger.warning(f'The embedding dim and model dim do not match ({pos_encoding_dim} vs {mha_dim}), adding a projector layer.')
                self.embedding_projector = nn.Linear(pos_encoding_dim, mha_dim)
        else:
            self.pose_encoding = None
        if config.residual_norm_style == ResidualNormStyle.DeepNorm:
            _, deep_norm_coefficients = get_deepnorm_coefficients(encoder_layers=0, decoder_layers=config.num_layers)
            assert deep_norm_coefficients is not None
            residual_scale = deep_norm_coefficients.alpha
        else:
            residual_scale = 1.0
        ln_factory = _get_ln_factory(config.dim_model, config.residual_norm_style, use_triton=config.use_triton, residual=True, residual_scale=residual_scale, normalization=config.normalization)
        mha = build_multi_head_attention(config.multi_head_config_masked)
        cross_mha = build_multi_head_attention(config.multi_head_config_cross)
        feedforward = build_feedforward(config.feedforward_config)
        self.supports_attention_mask = mha.attention.supports_attention_mask
        self.requires_same_k_q_dimensions = mha.attention.requires_same_k_q_dimensions
        self.requires_squared_context_length = feedforward.requires_squared_context or mha.attention.requires_squared_context
        self.causal_attention = mha.attention.causal if hasattr(mha.attention, 'causal') else False
        self.wrap_att = ln_factory(mha)
        self.wrap_cross = ln_factory(cross_mha)
        self.wrap_ff: Union[Residual, PostNorm] = ln_factory(feedforward)
        if config.residual_norm_style == ResidualNormStyle.Pre and config.layer_position.is_last():
            self.wrap_ff = PostNorm(config.dim_model, self.wrap_ff, normalization=NormalizationType.LayerNorm)

    @classmethod
    def from_config(cls, config: xFormerDecoderConfig):
        return cls(config)

    def forward(self, target: torch.Tensor, memory: torch.Tensor, encoder_att_mask: Optional[Union[torch.Tensor, AttentionMask]]=None, decoder_att_mask: Optional[Union[torch.Tensor, AttentionMask]]=None, input_mask: Optional[torch.Tensor]=None):
        if self.pose_encoding is not None:
            target = self.pose_encoding(target)
            if hasattr(self, 'embedding_projector'):
                target = self.embedding_projector(target)
        if input_mask is not None:
            target_q = target
            target_k = target * input_mask.unsqueeze(-1)
            target_v = target_k
        else:
            target_q, target_k, target_v = (target, target, target)
        x = self.wrap_att(inputs=[target_q, target_k, target_v], att_mask=decoder_att_mask)
        x = self.wrap_cross(inputs=[x, memory, memory], att_mask=encoder_att_mask)
        x = self.wrap_ff(inputs=[x])
        return x