import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from xformers.components.attention import (
from xformers.components.attention.core import _softmax
from xformers.components.input_projection import InputProjection, InputProjectionConfig
def _either_or(a: Optional[int], b: int) -> int:
    return a if a is not None else b