from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch
@dataclass
class FeatureMapConfig:
    name: str
    dim_features: int
    iter_before_redraw: Optional[int]
    normalize_inputs: Optional[bool]
    epsilon: Optional[float]