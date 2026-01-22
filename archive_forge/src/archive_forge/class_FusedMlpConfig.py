import logging
from dataclasses import dataclass
import torch
import torch.nn as nn
from xformers.components import Activation
from xformers.components.feedforward import (
@dataclass
class FusedMlpConfig(FeedforwardConfig):
    hidden_layer_multiplier: int