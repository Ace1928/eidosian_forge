from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch
import torch.nn as nn
from xformers.components.attention import AttentionMask
@dataclass
class AttentionConfig:
    """Parameters required for all Attentions.
    Can accept and store extra parameters.
    """
    name: str
    dropout: float