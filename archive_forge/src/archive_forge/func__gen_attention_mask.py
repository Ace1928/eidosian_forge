import math
from typing import List, Optional, Tuple
import torch
def _gen_attention_mask(self, input: torch.Tensor) -> torch.Tensor:
    utterance_length = input.size(0)
    num_segs = math.ceil(utterance_length / self.segment_length)
    rc_mask = []
    query_mask = []
    summary_mask = []
    if self.use_mem:
        num_cols = 9
        rc_q_cols_mask = [idx in [1, 4, 7] for idx in range(num_cols)]
        s_cols_mask = [idx in [4, 7] for idx in range(num_cols)]
        masks_to_concat = [rc_mask, query_mask, summary_mask]
    else:
        num_cols = 6
        rc_q_cols_mask = [idx in [1, 4] for idx in range(num_cols)]
        s_cols_mask = None
        masks_to_concat = [rc_mask, query_mask]
    for seg_idx in range(num_segs):
        col_widths = self._gen_attention_mask_col_widths(seg_idx, utterance_length)
        rc_mask_block = _gen_attention_mask_block(col_widths, rc_q_cols_mask, self.right_context_length, input.device)
        rc_mask.append(rc_mask_block)
        query_mask_block = _gen_attention_mask_block(col_widths, rc_q_cols_mask, min(self.segment_length, utterance_length - seg_idx * self.segment_length), input.device)
        query_mask.append(query_mask_block)
        if s_cols_mask is not None:
            summary_mask_block = _gen_attention_mask_block(col_widths, s_cols_mask, 1, input.device)
            summary_mask.append(summary_mask_block)
    attention_mask = (1 - torch.cat([torch.cat(mask) for mask in masks_to_concat])).to(torch.bool)
    return attention_mask