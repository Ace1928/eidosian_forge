from contextlib import contextmanager
import numpy as np
from_record_like = None
def array_core(ary):
    """
    Extract the repeated core of a broadcast array.

    Broadcast arrays are by definition non-contiguous due to repeated
    dimensions, i.e., dimensions with stride 0. In order to ascertain memory
    contiguity and copy the underlying data from such arrays, we must create
    a view without the repeated dimensions.

    """
    if not ary.strides or not ary.size:
        return ary
    core_index = []
    for stride in ary.strides:
        core_index.append(0 if stride == 0 else slice(None))
    return ary[tuple(core_index)]