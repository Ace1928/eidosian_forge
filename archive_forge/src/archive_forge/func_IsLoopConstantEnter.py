import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsLoopConstantEnter(op):
    """Return true iff op is a loop invariant."""
    return IsLoopEnter(op) and op.get_attr('is_constant')