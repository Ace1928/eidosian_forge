import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
def _compute_ema_coefficients(self):
    self._coeffs = None
    damping_factor = torch.sigmoid(self.damping_factor)
    decay_factor = torch.sigmoid(self.decay_factor)
    previous_timestep_weight = 1.0 - damping_factor * decay_factor
    return (damping_factor, previous_timestep_weight)