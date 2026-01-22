import math
from enum import Enum, auto
from typing import Optional
import torch
from torch.autograd.profiler import record_function
from .base import FeatureMap
class NormDistribution(Enum):
    Xi = auto()
    Uniform = auto()