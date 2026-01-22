import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsMerge(op):
    """Return true if `op` is a Merge."""
    return op.type == 'Merge' or op.type == 'RefMerge'