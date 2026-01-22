import math
from dataclasses import dataclass
from typing import (
import torch
class AttentionBias:
    """Base class for a custom bias that can be applied         as the attn_bias argument in
        :attr:`xformers.ops.memory_efficient_attention`.

    That function has the ability to add a tensor, the
    attention bias, to the QK^T matrix before it is used
    in the softmax part of the attention calculation.
    The attention bias tensor with shape
    (B or 1, n_queries, number of keys)
    can be given as the attn_bias input.
    The most common use case is for an attention bias is
    to contain only zeros and negative infinities, which forms
    a mask so that some queries only attend to some keys.

    Children of this class define alternative things which can
    be used as the attn_bias input to define an attention bias which
    forms such a mask, for some common cases.

    When using an :attr:`xformers.ops.AttentionBias`
    instead of a :attr:`torch.Tensor`, the mask matrix does
    not need to be materialized, and can be
    hardcoded into some kernels for better performance.

    See:

    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularMask`
    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularFromBottomRightMask`
    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularMaskWithTensorBias`
    - :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`
    - :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`

    """

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        """
        Materializes the bias as a `torch.Tensor`. This is very slow
        and we don't attempt to make it fast. Only use for debugging/testing.

        Shape should be like `[*, q_seqlen, k_seqlen]`
        """
        raise NotImplementedError()