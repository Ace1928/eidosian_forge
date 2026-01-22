from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from xformers.components import NormalizationType, ResidualNormStyle
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, FeedforwardConfig
from xformers.components.positional_embedding import (
from xformers.utils import generate_matching_config
@dataclass(init=False)
class xFormerBlockConfig:
    """
    The configuration structure to define a Transformer block.
    This base class is applicable to both encoder and decoder definitions.

    This completely defines each of the blocks, for instance in terms of dimensions,
    position encoding, pre or post layer norms or reversibility.
    """
    dim_model: int
    feedforward_config: FeedforwardConfig
    position_encoding_config: Optional[PositionEmbeddingConfig]
    block_type: BlockType
    residual_norm_style: ResidualNormStyle
    normalization: NormalizationType
    layer_position: LayerPosition
    use_triton: bool
    reversible: bool
    num_layers: int

    def __init__(self, dim_model: int, feedforward_config: Dict[str, Any], position_encoding_config: Optional[Dict[str, Any]], block_type: BlockType, residual_norm_style: ResidualNormStyle=ResidualNormStyle('post'), normalization: NormalizationType=NormalizationType.LayerNorm, reversible: bool=False, num_layers: int=1, layer_position: Optional[LayerPosition]=None):
        self.dim_model = dim_model
        self.block_type = block_type
        self.residual_norm_style = residual_norm_style
        self.reversible = reversible
        self.num_layers = num_layers
        self.normalization = normalization
        self.feedforward_config = generate_matching_config(feedforward_config, FEEDFORWARD_REGISTRY[feedforward_config['name']].config)
        self.position_encoding_config = generate_matching_config(position_encoding_config, POSITION_EMBEDDING_REGISTRY[position_encoding_config['name']].config) if position_encoding_config is not None else None
        if layer_position:
            self.layer_position = layer_position
        else:
            self.layer_position = LayerPosition()