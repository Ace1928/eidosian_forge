import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from xformers._deprecation_warning import deprecated_function
from xformers.components.residual import ResidualNormStyle
@dataclass
class BasicLayerConfig:
    embedding: int
    attention_mechanism: str
    patch_size: int
    stride: int
    padding: int
    seq_len: int
    feedforward: str
    normalization: str = 'layernorm'
    repeat_layer: int = 1