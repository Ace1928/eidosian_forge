from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def _take_slice(x, n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr, keep_dim: core.constexpr=True):
    y = sum(x * _indicator(n_dims, idx, pos), n_dims - 1 - idx)
    if keep_dim:
        y = core.expand_dims(y, n_dims - 1 - idx)
    return y