import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def add_op_consumer(self, src_op_name, src_slot, dst_op_name, dst_slot):
    """Add a consuming op for this op.

    Args:
      src_op_name: Name of the op of which the output tensor is being consumed.
      src_slot: 0-based output slot of the op being consumed.
      dst_op_name: Name of the consuming op (e.g., "Conv2D_3/BiasAdd")
      dst_slot: 0-based input slot of the consuming op that receives the tensor
        from this op.
    """
    self._op_consumers[src_op_name].append((src_slot, dst_op_name, dst_slot))