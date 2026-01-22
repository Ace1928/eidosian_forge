import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union
import torch
from xformers.components import Activation
from xformers.components.feedforward import (
@dataclass
class MoEConfig(FeedforwardConfig):
    number_of_experts: int
    gate: GateConfig
    number_of_local_experts: Optional[int] = None
    expert_constructor: Optional[Any] = None
    hidden_layer_multiplier: Optional[int] = None
    group: Optional[Any] = None