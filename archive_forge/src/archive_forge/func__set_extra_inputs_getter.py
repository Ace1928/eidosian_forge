from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def _set_extra_inputs_getter(self, extra_inputs_getter: Callable) -> BackendPatternConfig:
    self._extra_inputs_getter = extra_inputs_getter
    return self