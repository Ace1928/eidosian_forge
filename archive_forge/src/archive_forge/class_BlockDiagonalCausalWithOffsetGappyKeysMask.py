import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class BlockDiagonalCausalWithOffsetGappyKeysMask(BlockDiagonalGappyKeysMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`,
    except k/v is gappy.

    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is nearer to the final key in block i
    than Q is to the final query in block i.
    """

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        if shape[-1] != self.k_seqinfo.seqstart_py[-1]:
            raise ValueError('k shapes wrong')
        if shape[-2] != self.q_seqinfo.seqstart_py[-1]:
            raise ValueError('q shapes wrong')
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(self.q_seqinfo.intervals(), self.k_seqinfo.intervals())):
            mask[q_start:q_end, k_start:k_end] = LowerTriangularFromBottomRightMask().materialize(shape=(q_end - q_start, k_end - k_start), dtype=dtype, device=device)
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)