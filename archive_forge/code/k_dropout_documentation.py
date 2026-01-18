import triton
import triton.language as tl
from xformers.triton.k_activations import (

    Apply dropout on an input tensor
    GRAD_OUT    (M, N)
    GRAD_BIAS   (N,)
    GRAD_IN     (M, N)
    BIAS        (N,)
    SEEDS       (N,)
    p : dropout probability
    