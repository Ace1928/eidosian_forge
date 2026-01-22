import math
from dataclasses import dataclass
from typing import (
import torch
class LowerTriangularFromBottomRightMask(AttentionBias):
    """
    A causal masking.

    This mask is exactly the same as :attr:`LowerTriangularMask` when there is
    the same number of queries and keys.
    When the number of queries is different from the number of keys,
    it is a triangular mask shifted so that the last query can attend to
    the last key.
    In other words, a query Q cannot attend to a key which is nearer the
    final key than Q is to the final query.


    .. figure:: /_static/causal_bottom_right.png

        The difference between :attr:`LowerTriangularMask` (left) and
        :attr:`LowerTriangularFromBottomRightMask` (right). They become
        equivalent if the number of queries equals the number of keys.
    """

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        return _materialize_causal_mask(shape, dtype=dtype, device=device, from_bottomright=True)

    def make_local_attention(self, window_size: int) -> 'LowerTriangularFromBottomRightLocalAttentionMask':
        """
        Create a new bias which combines local + causal attention.

        See :attr:`LowerTriangularFromBottomRightLocalAttentionMask`
        """
        return LowerTriangularFromBottomRightLocalAttentionMask(window_size)