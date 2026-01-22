import math
from typing import List, Optional, Tuple
import torch
def _gen_key_value(self, input: torch.Tensor, mems: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    T, _, _ = input.shape
    summary_length = mems.size(0) + 1
    right_ctx_utterance_block = input[:T - summary_length]
    mems_right_ctx_utterance_block = torch.cat([mems, right_ctx_utterance_block])
    key, value = self.emb_to_key_value(mems_right_ctx_utterance_block).chunk(chunks=2, dim=2)
    return (key, value)