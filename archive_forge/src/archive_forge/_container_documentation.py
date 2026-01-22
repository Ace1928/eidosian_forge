from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import torch
from torch import nn
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import Transform
[BETA] Apply a list of transformations in a random order.

    .. v2betastatus:: RandomOrder transform

    This transform does not support torchscript.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
    