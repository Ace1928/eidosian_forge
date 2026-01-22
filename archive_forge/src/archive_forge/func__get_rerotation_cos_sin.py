from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
def _get_rerotation_cos_sin(self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if key_states.shape[-2] not in self.cos_sin_cache:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
        original_cos = cos[self.num_sink_tokens + key_states.shape[-2]:]
        shifted_cos = cos[self.num_sink_tokens:-key_states.shape[-2]]
        original_sin = sin[self.num_sink_tokens + key_states.shape[-2]:]
        shifted_sin = sin[self.num_sink_tokens:-key_states.shape[-2]]
        rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
        rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin
        self.cos_sin_cache[key_states.shape[-2]] = (rerotation_cos.to(key_states.dtype).unsqueeze(0), rerotation_sin.to(key_states.dtype).unsqueeze(0))
    return self.cos_sin_cache[key_states.shape[-2]]