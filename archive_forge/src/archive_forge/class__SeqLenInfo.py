import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class _SeqLenInfo:
    """
    (Internal) Represents the division of a dimension into blocks.

    For example, to represents a dimension of length 7 divided into
    three blocks of lengths 2, 3 and 2, use `from_seqlength([2, 3, 2])`.
    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 2, 5, 7]
        seqstart: torch.IntTensor([0, 2, 5, 7])
    """
    seqstart: torch.Tensor
    max_seqlen: int
    min_seqlen: int
    seqstart_py: List[int]

    def to(self, device: torch.device) -> None:
        self.seqstart = self.seqstart.to(device, non_blocking=True)

    def intervals(self) -> Iterable[Tuple[int, int]]:
        yield from zip(self.seqstart_py, self.seqstart_py[1:])

    @classmethod
    def _get_seqstart(cls, seqlens: Iterable[int]) -> Tuple[int, int, List[int], torch.Tensor]:
        """
        Given sequence lengths, returns the min/max value and the sequence start
        positions (offsets), with first element being 0 (returned in list and Tensor).
        """
        assert not isinstance(seqlens, torch.Tensor)
        seqstart_py = [0]
        max_seqlen = -1
        min_seqlen = -1
        for seqlen in seqlens:
            min_seqlen = min(min_seqlen, seqlen) if min_seqlen != -1 else seqlen
            max_seqlen = max(max_seqlen, seqlen)
            seqstart_py.append(seqstart_py[len(seqstart_py) - 1] + seqlen)
        seqstart = torch.tensor(seqstart_py, dtype=torch.int32)
        return (min_seqlen, max_seqlen, seqstart_py, seqstart)

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> '_SeqLenInfo':
        """
        Input tensors are assumed to be in shape [B, M, *]
        """
        min_seqlen, max_seqlen, seqstart_py, seqstart = cls._get_seqstart(seqlens)
        return cls(max_seqlen=max_seqlen, min_seqlen=min_seqlen, seqstart=seqstart, seqstart_py=seqstart_py)

    def split(self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]]=None) -> List[torch.Tensor]:
        if self.seqstart_py[-1] != x.shape[1] or x.shape[0] != 1:
            raise ValueError(f'Invalid `torch.Tensor` of shape {x.shape}, expected format (B, M, *) with B=1 and M={self.seqstart_py[-1]}\n seqstart: {self.seqstart_py}')
        if batch_sizes is None:
            batch_sizes = [1] * (len(self.seqstart_py) - 1)
        split_chunks = []
        it = 0
        for batch_size in batch_sizes:
            split_chunks.append(self.seqstart_py[it + batch_size] - self.seqstart_py[it])
            it += batch_size
        return [tensor.reshape([bs, -1, *tensor.shape[2:]]) for bs, tensor in zip(batch_sizes, x.split(split_chunks, dim=1))]