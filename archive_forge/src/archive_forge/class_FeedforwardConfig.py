from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch.nn as nn
from xformers.components import Activation
@dataclass
class FeedforwardConfig:
    name: str
    dim_model: int
    dropout: float
    activation: Activation