import math
from typing import List, Optional, Tuple
import torch
def _gen_attention_mask_col_widths(self, seg_idx: int, utterance_length: int) -> List[int]:
    num_segs = math.ceil(utterance_length / self.segment_length)
    rc = self.right_context_length
    lc = self.left_context_length
    rc_start = seg_idx * rc
    rc_end = rc_start + rc
    seg_start = max(seg_idx * self.segment_length - lc, 0)
    seg_end = min((seg_idx + 1) * self.segment_length, utterance_length)
    rc_length = self.right_context_length * num_segs
    if self.use_mem:
        m_start = max(seg_idx - self.max_memory_size, 0)
        mem_length = num_segs - 1
        col_widths = [m_start, seg_idx - m_start, mem_length - seg_idx, rc_start, rc, rc_length - rc_end, seg_start, seg_end - seg_start, utterance_length - seg_end]
    else:
        col_widths = [rc_start, rc, rc_length - rc_end, seg_start, seg_end - seg_start, utterance_length - seg_end]
    return col_widths