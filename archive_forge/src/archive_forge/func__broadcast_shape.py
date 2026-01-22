import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch, set_module
def _broadcast_shape(*args):
    """Returns the shape of the arrays that would result from broadcasting the
    supplied arrays against each other.
    """
    b = np.broadcast(*args[:32])
    for pos in range(32, len(args), 31):
        b = broadcast_to(0, b.shape)
        b = np.broadcast(b, *args[pos:pos + 31])
    return b.shape