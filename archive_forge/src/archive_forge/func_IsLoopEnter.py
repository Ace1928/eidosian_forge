import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsLoopEnter(op):
    """Returns true if `op` is an Enter."""
    return op.type == 'Enter' or op.type == 'RefEnter'