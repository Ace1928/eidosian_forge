from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch.nn as nn
from xformers.components import Activation
class Feedforward(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, dim_model: Optional[int]=None, dropout: Optional[float]=None, activation: Optional[Activation]=None, *args, **kwargs):
        super().__init__()
        self.requires_cuda = False
        self.requires_squared_context = False

    @classmethod
    def from_config(cls: Type[Self], config: FeedforwardConfig) -> Self:
        fields = asdict(config)
        fields = {k: v for k, v in fields.items() if v is not None}
        return cls(**fields)