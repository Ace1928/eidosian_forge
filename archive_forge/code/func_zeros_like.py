from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def zeros_like(input):
    return zeros(input.shape, input.dtype)