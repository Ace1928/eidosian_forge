from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from xformers.components import NormalizationType, ResidualNormStyle
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, FeedforwardConfig
from xformers.components.positional_embedding import (
from xformers.utils import generate_matching_config
class LayerPositionBitmask(int, Enum):
    First = 1
    Last = 2
    Default = 3