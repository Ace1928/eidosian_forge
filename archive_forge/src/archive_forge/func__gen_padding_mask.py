import math
from typing import List, Optional, Tuple
import torch
def _gen_padding_mask(utterance: torch.Tensor, right_context: torch.Tensor, summary: torch.Tensor, lengths: torch.Tensor, mems: torch.Tensor, left_context_key: Optional[torch.Tensor]=None) -> Optional[torch.Tensor]:
    T = right_context.size(0) + utterance.size(0) + summary.size(0)
    B = right_context.size(1)
    if B == 1:
        padding_mask = None
    else:
        right_context_blocks_length = T - torch.max(lengths).int() - summary.size(0)
        left_context_blocks_length = left_context_key.size(0) if left_context_key is not None else 0
        klengths = lengths + mems.size(0) + right_context_blocks_length + left_context_blocks_length
        padding_mask = _lengths_to_padding_mask(lengths=klengths)
    return padding_mask