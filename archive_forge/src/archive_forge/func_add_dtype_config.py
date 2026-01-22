from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def add_dtype_config(self, dtype_config: DTypeConfig) -> BackendPatternConfig:
    """
        Add a set of supported data types passed as arguments to quantize ops in the
        reference model spec.
        """
    self.dtype_configs.append(dtype_config)
    return self