from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from xformers.components import NormalizationType, ResidualNormStyle
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, FeedforwardConfig
from xformers.components.positional_embedding import (
from xformers.utils import generate_matching_config
@dataclass(init=False)
class xFormerDecoderConfig(xFormerBlockConfig):
    """
    The configuration structure for a decoder block.

    This specifically defines the masked and cross attention mechanisms,
    on top of the settings defining all blocks.
    """
    multi_head_config_masked: Dict[str, Any]
    multi_head_config_cross: Dict[str, Any]

    def __init__(self, dim_model: int, feedforward_config: Dict[str, Any], multi_head_config_masked: Dict[str, Any], multi_head_config_cross: Dict[str, Any], position_encoding_config: Optional[Dict[str, Any]]=None, residual_norm_style: str='post', normalization: NormalizationType=NormalizationType.LayerNorm, use_triton: bool=True, **kwargs):
        try:
            if 'dim_model' not in multi_head_config_masked.keys():
                multi_head_config_masked['dim_model'] = dim_model
            if 'dim_model' not in multi_head_config_cross.keys():
                multi_head_config_cross['dim_model'] = dim_model
            if 'dim_model' not in feedforward_config.keys():
                feedforward_config['dim_model'] = dim_model
            if position_encoding_config is not None and 'dim_model' not in position_encoding_config.keys():
                position_encoding_config['dim_model'] = dim_model
        except AttributeError:
            pass
        if 'block_type' in kwargs.keys():
            assert kwargs['block_type'] == 'decoder'
        kwargs['block_type'] = BlockType('decoder')
        super().__init__(dim_model=dim_model, feedforward_config=feedforward_config, position_encoding_config=position_encoding_config, residual_norm_style=ResidualNormStyle(residual_norm_style), normalization=NormalizationType(normalization), **kwargs)
        self.multi_head_config_masked = multi_head_config_masked
        self.multi_head_config_cross = multi_head_config_cross
        self.use_triton = use_triton