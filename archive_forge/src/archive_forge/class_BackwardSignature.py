import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union
import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from .. import config
from .utils import strict_zip
@dataclass
class BackwardSignature:
    """
    Provides information about the backward section of an exported
    joint forward-backward graph.
    For a particular fx GraphModule, this class contains information on:
    (1) A mapping from each gradient (backwards output) to the parameter
        it corresponds to (forward input)
    (2) A mapping from each gradient (backwards output) to the user input
        it corresponds to (forward input)
    (3) Which of the forward outputs corresponds to the loss, that we backprop on.

    Each string name is the `node.name` of the corresponding node in the fx graph.
    """
    gradients_to_parameters: Dict[str, str]
    gradients_to_user_inputs: Dict[str, str]
    loss_output: str