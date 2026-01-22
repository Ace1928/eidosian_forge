import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class BlockDiagonalCausalLocalAttentionFromBottomRightMask(BlockDiagonalCausalFromBottomRightMask):
    """
    (Experimental feature)
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`.
    This makes the mask "local" and the attention pattern banded.

    Query i only attends to keys in its block and cannot attend keys further than "window_size"
    from it.
    """
    _window_size: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self._window_size <= 0:
            raise ValueError(f'Expected `window_size > 0`, but window_size={self._window_size}')

    def _create_block_mask(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        return _materialize_causal_mask(shape, dtype=dtype, device=device, window_size=self._window_size, from_bottomright=True)