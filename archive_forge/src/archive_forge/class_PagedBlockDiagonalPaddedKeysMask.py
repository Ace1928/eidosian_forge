import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class PagedBlockDiagonalPaddedKeysMask(AttentionBias):
    """
    Same as BlockDiagonalPaddedKeysMask, but for paged attention.
    block_tables has shape [batch_size, max_num_pages] and K/V have shape
    [1, max_num_pages * page_size, num_heads, head_dim]
    or [1, max_num_pages * page_size, num_groups, num_heads, head_dim]
    """
    q_seqinfo: _SeqLenInfo
    k_seqinfo: _PaddedSeqLenInfo
    block_tables: torch.Tensor
    page_size: int
    _UNPAGED_TYPE: ClassVar[Type[BlockDiagonalPaddedKeysMask]] = BlockDiagonalPaddedKeysMask

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        max_row_len = self.block_tables.shape[1] * self.page_size
        bias_nonpaged = self._UNPAGED_TYPE(q_seqinfo=self.q_seqinfo, k_seqinfo=_PaddedSeqLenInfo.from_seqlens_padded(self.k_seqinfo.seqlen_py, max_row_len))
        mask_nonpaged = bias_nonpaged.materialize(shape, dtype, device)
        n_used_blocks = cast(int, self.block_tables.max().item() + 1)
        max_physical_len = n_used_blocks * self.page_size
        mask_paged = torch.empty(mask_nonpaged.shape[:-1] + (max_physical_len,), dtype=dtype, device=device)
        mask_paged.fill_(-math.inf)
        for b, (q_start, q_end) in enumerate(self.q_seqinfo.intervals()):
            for logical_page_idx in range(self.block_tables.shape[1]):
                physical_page_idx = cast(int, self.block_tables[b][logical_page_idx].item())
                k_logical_start = b * max_row_len + logical_page_idx * self.page_size
                k_logical_end = k_logical_start + self.page_size
                k_physical_start = physical_page_idx * self.page_size
                k_physical_end = k_physical_start + self.page_size
                mask_paged[..., q_start:q_end, k_physical_start:k_physical_end] = mask_nonpaged[..., q_start:q_end, k_logical_start:k_logical_end]
        return mask_paged

    @classmethod
    def from_seqlens(cls, q_seqlen: Sequence[int], kv_seqlen: Sequence[int], block_tables: torch.Tensor, page_size: int) -> 'PagedBlockDiagonalPaddedKeysMask':
        """Creates a :attr:`PagedBlockDiagonalPaddedKeysMask` from a list of tensor
        lengths for query and key/value.

        Args:
            q_seqlen (Sequence[int]): List or tensor of sequence lengths for query tensors
            kv_padding (int): Padding for k/v - also an upperbound on each individual key length
            kv_seqlen (Sequence[int]): List or tensor of sequence lengths for key/value.
            causal_diagonal: unused, for BC only
        Returns:
            PagedBlockDiagonalPaddedKeysMask
        """
        assert len(q_seqlen) == len(kv_seqlen), (q_seqlen, kv_seqlen)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(kv_seqlen, padding=block_tables.shape[1] * page_size)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, block_tables=block_tables, page_size=page_size)