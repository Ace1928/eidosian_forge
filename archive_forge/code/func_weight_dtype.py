from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
@property
def weight_dtype(self) -> Optional[torch.dtype]:
    return self.weight_dtype_with_constraints.dtype