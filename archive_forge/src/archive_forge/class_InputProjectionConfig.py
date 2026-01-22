import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
@dataclass
class InputProjectionConfig:
    in_features: int
    out_features: int
    bias: bool