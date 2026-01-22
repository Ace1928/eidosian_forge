import math
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
from xformers.components import Activation, build_activation
from xformers.components.feedforward import Feedforward, FeedforwardConfig
from . import register_feedforward
@dataclass
class ConvMlpConfig(FeedforwardConfig):
    hidden_layer_multiplier: int
    dim_model: int
    dim_model_out: Optional[int]
    act_layer: Activation
    dropout: float