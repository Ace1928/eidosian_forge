import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class PagedBlockDiagonalCausalWithOffsetPaddedKeysMask(PagedBlockDiagonalPaddedKeysMask):
    """
    Same as BlockDiagonalCausalWithOffsetPaddedKeysMask, but for paged attention.
    block_tables has shape [batch_size, max_num_pages] and K/V have shape
    [1, max_num_pages * page_size, num_heads, head_dim]
    or [1, max_num_pages * page_size, num_groups, num_heads, head_dim]
    """
    _UNPAGED_TYPE = BlockDiagonalCausalWithOffsetPaddedKeysMask