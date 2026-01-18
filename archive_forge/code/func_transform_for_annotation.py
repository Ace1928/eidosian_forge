from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node
def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Allows for user defined transforms to run before annotating the graph.
        This allows quantizer to allow quantizing part of the model that are otherwise not quantizable.
        For example quantizer can
        a) decompose a compound operator like scaled dot product attention,
        into bmm and softmax if quantizer knows how to quantize bmm/softmax but not sdpa
        or b) transform scalars to tensor to allow quantizing scalares.

        Note: this is an optional method
        """
    return model