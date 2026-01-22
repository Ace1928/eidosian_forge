import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _sparse_coo_scatter_reduction_helper(op, mask_input: Tensor, dims: Tuple[int, ...], keepdim: bool, dtype: Optional[DType]=None) -> Tensor:
    reduce = op.__name__
    valid_reductions = ['sum', 'prod', 'amax', 'amin']
    if reduce not in valid_reductions:
        raise ValueError(f'op must be one of {' '.join(valid_reductions)}, but got {reduce} instead')
    output_dtype = dtype
    values, indices = (mask_input._values(), mask_input._indices())
    input_dims = mask_input.dim()
    num_sparse_dims = mask_input.sparse_dim()
    reduced_sparse_dims = []
    retained_sparse_dims = []
    reduced_dense_dims = []
    if values.dtype != output_dtype:
        values = values.to(output_dtype)
    if keepdim:
        output_shape = tuple((1 if i in dims else si for i, si in enumerate(mask_input.shape)))
    else:
        output_shape = tuple((si for i, si in enumerate(mask_input.shape) if i not in dims))
    for d in dims:
        if d >= input_dims:
            continue
        if d < num_sparse_dims:
            reduced_sparse_dims.append(d)
        else:
            reduced_dense_dims.append(d + 1 - num_sparse_dims)
    if len(reduced_dense_dims) > 0:
        if reduce == 'sum':
            new_values = values
            new_values = op(new_values, dim=reduced_dense_dims, keepdim=bool(keepdim))
        else:
            return NotImplemented
    else:
        new_values = values.clone()
    if len(reduced_sparse_dims) == num_sparse_dims:
        if reduce in {'amax', 'amin'} and new_values.size(0) == 0:
            new_values = _reduction_identity(reduce, new_values)
        else:
            new_values = op(new_values, dim=0)
        if keepdim:
            for _ in range(num_sparse_dims):
                new_values = new_values.unsqueeze(0)
        return new_values.to(dtype=output_dtype).to_sparse()
    else:
        new_indices = indices.clone()
        if keepdim:
            new_indices[reduced_sparse_dims, :] = 0
        elif len(reduced_sparse_dims) > 0:
            retained_sparse_dims = [i for i in range(num_sparse_dims) if i not in set(reduced_sparse_dims)]
            new_indices = new_indices.index_select(0, torch.tensor(retained_sparse_dims).to(mask_input.device))
    if new_indices.numel() > 0:
        new_indices, inverse_indices = torch.unique(new_indices, return_inverse=True, dim=1)
        out_shape = list(new_values.shape)
        out_shape[0] = new_indices.shape[1]
        for _ in range(new_values.ndim - 1):
            inverse_indices = inverse_indices.unsqueeze(-1)
        scatter_indices = inverse_indices.expand(new_values.shape)
        if output_dtype in {torch.bfloat16, torch.float16}:
            new_values = new_values.to(torch.float)
            out = new_values.new_empty(out_shape)
            new_values = out.scatter_reduce_(0, scatter_indices, new_values, reduce=reduce, include_self=False)
            new_values = new_values.to(dtype=output_dtype)
        else:
            out = new_values.new_empty(out_shape)
            new_values = out.scatter_reduce_(0, scatter_indices, new_values, reduce=reduce, include_self=False)
    return torch.sparse_coo_tensor(new_indices, new_values, output_shape, dtype=output_dtype, device=mask_input.device)