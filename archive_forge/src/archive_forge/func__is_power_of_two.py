from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
def _is_power_of_two(i: core.constexpr):
    n = i.value
    return core.constexpr(n & n - 1 == 0 and n != 0)