import math
from typing import List, Optional, Tuple
import torch
def _apply_attention_infer(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, mems: torch.Tensor, state: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    if state is None:
        state = self._init_state(utterance.size(1), device=utterance.device)
    pre_mems, lc_key, lc_val = self._unpack_state(state)
    if self.use_mem:
        summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
        summary = summary[:1]
    else:
        summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
    rc_output, next_m, next_k, next_v = self.attention.infer(utterance=utterance, lengths=lengths, right_context=right_context, summary=summary, mems=pre_mems, left_context_key=lc_key, left_context_val=lc_val)
    state = self._pack_state(next_k, next_v, utterance.size(0), mems, state)
    return (rc_output, next_m, state)