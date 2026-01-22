from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node
@dataclass
class QuantizationAnnotation:
    """How are input arguemnt or output should be quantized,
    expressed as QuantizationSpec, this corresponds to how a Tensor in the
    operator Graph is observed (PTQ) or fake quantized (QAT)
    """
    input_qspec_map: Dict[Node, Optional[QuantizationSpecBase]] = field(default_factory=dict)
    output_qspec: Optional[QuantizationSpecBase] = None
    allow_implicit_sharing: bool = True
    _annotated: bool = False