from typing import Callable, List, Optional, Tuple, Union
import math
import warnings
import importlib
import torch
from torch import _VF
from torch import sym_int as _sym_int
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes, sparse_support_notes
from typing import TYPE_CHECKING
from .._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from ..overrides import (
from . import _reduction as _Reduction
from . import grad  # noqa: F401
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    if query.dim() == 3:
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, f'For batched (3-D) `query`, expected `key` and `value` to be 3-D but found {key.dim()}-D and {value.dim()}-D tensors respectively'
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, f'For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D but found {key_padding_mask.dim()}-D tensor instead'
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), f'For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D but found {attn_mask.dim()}-D tensor instead'
    elif query.dim() == 2:
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, f'For unbatched (2-D) `query`, expected `key` and `value` to be 2-D but found {key.dim()}-D and {value.dim()}-D tensors respectively'
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, f'For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D but found {key_padding_mask.dim()}-D tensor instead'
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), f'For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D but found {attn_mask.dim()}-D tensor instead'
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, f'Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}'
    else:
        raise AssertionError(f'query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor')
    return is_batched