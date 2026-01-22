import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _reduction_identity(op_name: str, input: Tensor, *args):
    """Return identity value as scalar tensor of a reduction operation on
    given input, or None, if the identity value cannot be uniquely
    defined for the given input.

    The identity value of the operation is defined as the initial
    value to reduction operation that has a property ``op(op_identity,
    value) == value`` for any value in the domain of the operation.
    Or put it another way, including or excluding the identity value in
    a list of operands will not change the reduction result.

    See https://github.com/pytorch/rfcs/pull/27 for more information.

    """
    dtype: DType = input.dtype
    device = input.device
    op_name = op_name.rsplit('.', 1)[-1]
    if op_name in {'sum', 'cumsum'}:
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name in {'prod', 'cumprod'}:
        return torch.tensor(1, dtype=dtype, device=device)
    elif op_name in {'amax', 'argmax', 'logsumexp'}:
        if torch.is_floating_point(input):
            return torch.tensor(-torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).min, dtype=dtype, device=device)
    elif op_name in {'amin', 'argmin'}:
        if torch.is_floating_point(input):
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).max, dtype=dtype, device=device)
    elif op_name == 'mean':
        return None
    elif op_name == 'norm':
        ord = args[0] if args else 2
        if ord == float('-inf'):
            assert torch.is_floating_point(input), input.dtype
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == 'median':
        dtype = input.dtype if torch.is_floating_point(input) else torch.float
        return torch.tensor(torch.nan, dtype=dtype, device=device)
    elif op_name in {'var', 'std'}:
        return None
    raise NotImplementedError(f'identity of {op_name} on {dtype} input')