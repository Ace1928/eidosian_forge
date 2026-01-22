from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node
@dataclass(eq=True, frozen=True)
class SharedQuantizationSpec(QuantizationSpecBase):
    """
    Quantization spec for the Tensors whose quantization parameters are shared with other Tensors
    """
    edge_or_node: EdgeOrNode