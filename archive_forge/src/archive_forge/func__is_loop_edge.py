import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def _is_loop_edge(op):
    """Returns true if the op is the end of a while-loop creating a cycle."""
    return op.type in ['NextIteration']