from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilenet_v2 import MobileNetV2Config
def apply_depth_multiplier(config: MobileNetV2Config, channels: int) -> int:
    return make_divisible(int(round(channels * config.depth_multiplier)), config.depth_divisible_by, config.min_depth)