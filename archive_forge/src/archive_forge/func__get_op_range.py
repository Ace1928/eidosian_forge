import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _get_op_range(self):
    """Sets the index range of the Ops that we will consider tracing."""
    found, op_range = self.get_flag_value(FLAG_NAME_OP_RANGE)
    if not found or not op_range:
        op_range = (-1, -1)
        return op_range
    match = _OP_RANGE_PAT.match(op_range)
    if not match:
        op_range = (-1, -1)
        return op_range
    op_range = (int(match.group(1)), int(match.group(2)))
    return op_range