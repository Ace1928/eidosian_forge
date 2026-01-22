from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def _set_pattern_complex_format(self, pattern: Pattern) -> BackendPatternConfig:
    """
        Set the pattern to configure, using the reversed nested tuple format.

        See the BackendConfig README for more detail:
        https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md#advanced-pattern-specification
        """
    if self.pattern is not None:
        raise ValueError("Only one of 'pattern' or 'pattern_complex_format' can be set")
    self._pattern_complex_format = pattern
    return self