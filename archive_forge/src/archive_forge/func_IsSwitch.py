import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsSwitch(op):
    """Return true if `op` is a Switch."""
    return op.type == 'Switch' or op.type == 'RefSwitch'