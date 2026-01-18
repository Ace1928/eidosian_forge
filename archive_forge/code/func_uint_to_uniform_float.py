from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def uint_to_uniform_float(x):
    """
    Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
    """
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        x = x.to(tl.int32, bitcast=True)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64))
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, -x - 1, x)
    return x * scale