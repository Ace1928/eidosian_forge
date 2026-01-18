from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def zeros(shape, dtype):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    return core.full(shape, 0, dtype)