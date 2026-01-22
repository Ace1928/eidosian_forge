from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
def _raise_if_incorrect_dtype(self, target_probs: torch.Tensor, bonus_token_ids: torch.Tensor, draft_probs: torch.Tensor, draft_token_ids: torch.Tensor) -> None:
    assert all((probs.dtype == self.probs_dtype for probs in [target_probs, draft_probs]))
    assert all((token_ids.dtype == self.token_id_dtype for token_ids in [bonus_token_ids, draft_token_ids]))