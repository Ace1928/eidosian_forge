from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
def _GetBcastSubshape(subscripts):
    """Returns a tuple denoting the slice mapping to ellipsis.

    For a given subscript, returns a tuple (start, end) denoting the start
    axis index and the (negative) end axis index respectively. For any input
    Tensor `x` described by the subscript, `x[start:end]` would be the slice
    represented by the ellipsis. E.g. For `ab...cd` returns `[1, -2]`.

    If ellipsis is not present in `subscripts`, returns `(0, 0)`.

    Args:
      subscripts: A string denoting the einsum subscript.
    """
    start = subscripts.find(ellipsis)
    if start == -1:
        return (0, 0)
    remaining = len(subscripts) - (start + len(ellipsis))
    end = -remaining if remaining > 0 else None
    return (start, end)