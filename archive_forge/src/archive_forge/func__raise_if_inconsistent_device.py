from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
def _raise_if_inconsistent_device(self, target_probs: torch.Tensor, bonus_token_ids: torch.Tensor, draft_probs: torch.Tensor, draft_token_ids: torch.Tensor) -> None:
    devices = [t.device for t in [target_probs, bonus_token_ids, draft_probs, draft_token_ids]]
    assert all([devices[0] == device for device in devices])