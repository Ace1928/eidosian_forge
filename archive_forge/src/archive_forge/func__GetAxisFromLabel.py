from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
def _GetAxisFromLabel(subscripts, label):
    """Returns the axis (possibly negative) corresponding to a label.

    Returns the axis index of the axis label if it is before an ellipsis (or if
    the ellipsis is not present), and the negative index if it occurs after the
    ellipsis. E.g. index of `b` in `ab...cd`, is `1`, but that of `c` is `-2`.

    For multiple occurrences, returns the leftmost one. If not found, returns
    None.

    Args:
      subscripts: A string denoting the einsum subscript (e.g. `ab...cd`)
      label: The single character axis label.
    """
    splits = subscripts.split(ellipsis)
    index = splits[0].find(label)
    if index != -1:
        return index
    if len(splits) < 2:
        return None
    index = splits[1].find(label)
    if index != -1:
        return index - len(splits[1])
    return None