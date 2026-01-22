import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _input_mask(input: Union[Tensor, MaskedTensor], *args, **kwargs) -> Tensor:
    """Return canonical input mask.

    A canonical input mask is defined as a boolean mask tensor that
    shape and layout matches with the shape and the layout of the
    input.

    The canonical input mask is computed from the :attr:`mask` tensor
    content to meet the following criteria:

    1. The shape of the canonical input mask is the same as the shape
       of :attr:`input` tensor. If the mask tensor has a smaller shape
       than the shape of the :attr:`input`, broadcasting rules will be
       applied. Downcasting of mask is not supported.

    2. The layout of the canonical input mask is the same as the
       layout of the :attr:`input` tensor. If the mask has different
       layout, it will be converted to the expected layout.  In the
       case of sparse COO layout, the canonical input mask will be
       coalesced.

    3. The dtype of the canonical input mask is torch.bool. If the
       mask dtype is not bool then it will be converted to bool dtype
       using `.to(dtype=bool)` method call.

    4. The elements of the canonical input mask have boolean values
       copied from the content of the :attr:`mask` tensor (after
       possible broadcasting and dtype conversion transforms).  In
       general, the sparsity pattern of the sparse canonical input
       mask need not to be the same as the sparsity pattern of the
       sparse :attr:`input` tensor.

    """
    if input.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
        raise ValueError(f'_input_mask expects strided or sparse COO or sparse CSR tensor but got {input.layout}')
    mask = kwargs.get('mask')
    if mask is None:
        raise ValueError('_input_mask requires explicit mask')
    if mask.shape != input.shape:
        if mask.ndim > input.ndim:
            raise IndexError('_input_mask expected broadcastable mask (got mask dimensionality higher than of the input)')
        if mask.layout == torch.strided:
            mask = torch.broadcast_to(mask.clone(), input.shape).to(dtype=torch.bool)
        elif mask.layout == torch.sparse_coo:
            mask = torch._sparse_broadcast_to(mask, input.shape)
        else:
            assert mask.layout == torch.sparse_csr
            mask = torch._sparse_broadcast_to(mask.to_sparse(), input.shape).to_sparse_csr()
    if mask.layout != input.layout:
        if input.layout == torch.strided:
            mask = mask.to_dense()
        elif input.layout == torch.sparse_coo:
            if mask.layout == torch.strided:
                mask = mask.to_sparse(input.sparse_dim())
            else:
                mask = mask.to_sparse()
        else:
            assert input.layout == torch.sparse_csr
            mask = mask.to_sparse_csr()
    if mask.layout == torch.sparse_coo:
        mask = mask.coalesce()
    mask = mask.to(dtype=torch.bool)
    return mask