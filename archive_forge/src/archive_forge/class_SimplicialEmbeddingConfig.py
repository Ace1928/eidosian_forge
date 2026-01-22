from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch
from xformers import _is_triton_available
@dataclass
class SimplicialEmbeddingConfig:
    L: int
    temperature: float