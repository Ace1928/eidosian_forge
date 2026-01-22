import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
@dataclass
class Gradients:
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    db: Optional[torch.Tensor] = None