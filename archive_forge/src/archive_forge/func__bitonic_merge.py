from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def _bitonic_merge(x, n_dims: core.constexpr, active_dims: core.constexpr, order_type: core.constexpr):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    core.static_assert(active_dims <= n_dims)
    if order_type == 2:
        desc_mask = _indicator(n_dims, active_dims, 1)
    else:
        desc_mask = order_type
    for i in core.static_range(active_dims):
        x = _compare_and_swap(x, desc_mask, n_dims, active_dims - 1 - i)
    return x