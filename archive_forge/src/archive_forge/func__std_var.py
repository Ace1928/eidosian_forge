import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _std_var(input: Union[Tensor, MaskedTensor], dim: DimOrDims, unbiased: Optional[bool], *, correction_opt: Optional[Union[int, float]], keepdim: Optional[bool], dtype: Optional[DType], mask: Optional[Tensor], take_sqrt: Optional[bool]) -> Tensor:
    assert unbiased is None or correction_opt is None, 'Only one of unbiased and correction may be given'
    correction = 1.0
    if unbiased is not None:
        correction = 1.0 if unbiased else 0.0
    if correction_opt is not None:
        correction = sym_float(correction_opt)
    if dtype is None:
        dtype = input.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.float32
    compute_dtype = dtype
    if not (compute_dtype.is_floating_point or compute_dtype.is_complex):
        compute_dtype = torch.float32
    if input.layout == torch.strided:
        if mask is None:
            count = sum(torch.ones(input.shape, dtype=torch.int64, device=input.device), dim, keepdim=True)
            sample_total = sum(input, dim, keepdim=True, dtype=dtype)
        else:
            inmask = _input_mask(input, mask=mask)
            count = sum(inmask.new_ones(input.shape, dtype=torch.int64), dim, keepdim=True, mask=inmask)
            sample_total = sum(input, dim, keepdim=True, dtype=dtype, mask=inmask)
        sample_mean = torch.divide(sample_total, count)
        x = torch.subtract(input, sample_mean)
        if mask is None:
            total = sum(x * x.conj(), dim, keepdim=keepdim, dtype=compute_dtype)
        else:
            total = sum(x * x.conj(), dim, keepdim=keepdim, dtype=compute_dtype, mask=inmask)
        if not keepdim:
            count = count.reshape(total.shape)
        if correction != 0:
            real_dtype = corresponding_real_dtype(compute_dtype) if compute_dtype.is_complex else compute_dtype
            count = count.to(real_dtype)
            count = torch.subtract(count, correction)
            count = torch.maximum(count, count.new_zeros([]))
        output = torch.divide(total, count).to(dtype=dtype)
        if take_sqrt:
            output = torch.sqrt(output)
        return output
    else:
        raise ValueError(f'masked std/var expects strided tensor (got {input.layout} tensor)')