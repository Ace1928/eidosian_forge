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
def _in_projection(q: Tensor, k: Tensor, v: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor, b_q: Optional[Tensor]=None, b_k: Optional[Tensor]=None, b_v: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
    """Perform the in-projection step of the attention operation.

    This is simply a triple of linear projections,
    with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = (q.size(-1), k.size(-1), v.size(-1))
    assert w_q.shape == (Eq, Eq), f'expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}'
    assert w_k.shape == (Eq, Ek), f'expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}'
    assert w_v.shape == (Eq, Ev), f'expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}'
    assert b_q is None or b_q.shape == (Eq,), f'expecting query bias shape of {(Eq,)}, but got {b_q.shape}'
    assert b_k is None or b_k.shape == (Eq,), f'expecting key bias shape of {(Eq,)}, but got {b_k.shape}'
    assert b_v is None or b_v.shape == (Eq,), f'expecting value bias shape of {(Eq,)}, but got {b_v.shape}'
    return (linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v))