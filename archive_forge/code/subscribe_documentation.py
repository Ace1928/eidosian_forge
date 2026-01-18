import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
Return the control outputs for a given op.

    Args:
      op: The op to fetch control outputs for.

    Returns:
      Iterable of control output ops.
    