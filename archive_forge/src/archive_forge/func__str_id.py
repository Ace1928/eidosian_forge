import os
import sys
from tensorflow.core.profiler import tfprof_log_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.profiler.internal import flops_registry  # pylint: disable=unused-import
from tensorflow.python.util.tf_export import tf_export
def _str_id(s, str_to_id):
    """Maps string to id."""
    num = str_to_id.get(s, None)
    if num is None:
        num = len(str_to_id)
        str_to_id[s] = num
    return num