import math
from typing import List, Optional, Tuple
import torch
def _apply_attention_forward(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, mems: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if attention_mask is None:
        raise ValueError('attention_mask must be not None when for_inference is False')
    if self.use_mem:
        summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
    else:
        summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
    rc_output, next_m = self.attention(utterance=utterance, lengths=lengths, right_context=right_context, summary=summary, mems=mems, attention_mask=attention_mask)
    return (rc_output, next_m)