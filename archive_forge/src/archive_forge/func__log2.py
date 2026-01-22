from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
def _log2(i: core.constexpr):
    log2 = 0
    n = i.value
    while n > 1:
        n >>= 1
        log2 += 1
    return core.constexpr(log2)