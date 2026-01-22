import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
@staticmethod
def _generate_value(left: float, right: float) -> float:
    return torch.empty(1).uniform_(left, right).item()