import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, device, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import (
from ...utils import logging
from .configuration_blip import BlipTextConfig
class BlipTextOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BlipTextLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores