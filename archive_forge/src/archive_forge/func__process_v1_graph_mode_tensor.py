import atexit
import os
import re
import socket
import threading
import uuid
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _process_v1_graph_mode_tensor(self, op_type, tensor, debug_tensor, tensor_debug_mode):
    """For V1 graph mode, determine what tensor to output from callback.

    Args:
      op_type: Type of the op that outputs the original symbolic tensor.
      tensor: The original output symbolic tensor.
      debug_tensor: The debugger-instrumented tensor.
      tensor_debug_mode: Debug mode used, a tfdbg TensorDebugMode enum.

    Returns:
      A symbolic tensor to be returned by the dumping op_callback.
    """
    if op_type in ('Placeholder', 'PlaceholderWithDefault'):
        self._placeholder_to_debug_tensor[tensor] = debug_tensor
        return tensor
    elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR and op_type != 'Const':
        self._tensor_aliases[debug_tensor.name] = tensor.name
        return debug_tensor
    else:
        with self._symbolic_tensor_counter_lock:
            identity_name = 'tfdbg_identity_%d' % self._symbolic_tensor_counter
        identity = array_ops.identity(tensor, name=identity_name)
        identity.op._add_control_input(debug_tensor.op)
        self._tensor_aliases[identity.name] = tensor.name
        return identity