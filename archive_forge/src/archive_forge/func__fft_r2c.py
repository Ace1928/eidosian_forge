import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _fft_r2c(func_name: str, input: TensorLikeType, n: Optional[int], dim: int, norm: NormType, forward: bool, onesided: bool) -> TensorLikeType:
    """Common code for performing any real to complex FFT (rfft or ihfft)"""
    torch._check(not input.dtype.is_complex, lambda: f'{func_name} expects a floating point input tensor, but got {input.dtype}')
    input = _maybe_promote_tensor_fft(input)
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    dim_size = n if n is not None else input.shape[dim]
    torch._check(dim_size >= 1, lambda: f'Invalid number of data points ({dim_size}) specified')
    if n is not None:
        input = _resize_fft_input(input, dims, (n,))
    ret = prims.fft_r2c(input, dim=dims, onesided=onesided)
    ret = _apply_norm(ret, norm, dim_size, forward)
    return ret if forward else torch.conj(ret)