import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class _PaddedSeqLenInfo(_SeqLenInfo):
    """
    (Internal)  Represents the division of a dimension into blocks which are
    padded out to the same total length.

    For example, to represent a dimension of length 12 with space for
    three blocks of length 4, but where the occupied lengths are
    2, 3 and 2, use `from_seqlens_padded([2, 3, 2], 4)`.

    The layout along the dimension is

     0 ─►  block 0
           block 0
           <space>
           <space>
     4 ─►  block 1
           block 1
           block 1
           <space>
     8 ─►  block 2
           block 2
           <space>
           <space>
    12 ─►

    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 4, 8, 12]
        seqstart: torch.IntTensor([0, 4, 8, 12])
        seqlen_py: [2, 3, 2]
        seqlen: torch.IntTensor([2, 3, 2])
        padding: 4
    """
    seqlen: torch.Tensor
    seqlen_py: Sequence[int]
    padding: int

    def __post_init__(self) -> None:
        assert len(self.seqstart_py) == len(self.seqlen_py) + 1

    def to(self, device: torch.device) -> None:
        self.seqlen = self.seqlen.to(device, non_blocking=True)
        super().to(device)

    def intervals(self) -> Iterable[Tuple[int, int]]:
        for (start, _), length in zip(super().intervals(), self.seqlen_py):
            yield (start, start + length)

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> '_SeqLenInfo':
        raise RuntimeError('Use either `_SeqLenInfo.from_seqlens` or `_PaddedSeqLenInfo.from_seqlens_padded`')

    @classmethod
    def from_seqlens_padded(cls, seqlens: Sequence[int], padding: int) -> '_PaddedSeqLenInfo':
        """
        Input tensors are assumed to be in shape [B, M, *]
        seqstart = padding * torch.arange(batch_size)
        """
        assert not isinstance(seqlens, torch.Tensor)
        assert all((seqlen <= padding for seqlen in seqlens)), f'Seqlens {seqlens} Padding {padding}'
        seqstart_py = list(range(0, len(seqlens) * padding + 1, padding))
        seqlen = torch.tensor(seqlens, dtype=torch.int32)
        return cls(seqlen=seqlen, seqlen_py=seqlens, max_seqlen=max(seqlens), min_seqlen=min(seqlens), seqstart=torch.tensor(seqstart_py, dtype=torch.int32), seqstart_py=seqstart_py, padding=padding)

    def split(self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]]=None) -> List[torch.Tensor]:
        raise NotImplementedError('_PaddedSeqLenInfo.split')