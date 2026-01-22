from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
def _apply_key_rotary_pos_emb(self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotated_key_states = key_states * cos + self._rotate_half(key_states) * sin
    return rotated_key_states