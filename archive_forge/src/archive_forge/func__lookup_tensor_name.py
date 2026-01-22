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
def _lookup_tensor_name(self, tensor):
    """Look up the name of a graph tensor.

    This method maps the name of a debugger-generated Identity or
    DebugIdentityV2 tensor to the name of the original instrumented tensor,
    if `tensor` is such a debugger-created tensor.
    Otherwise, it returns the name of `tensor` as is.

    Args:
      tensor: The graph tensor to look up the name for.

    Returns:
      Name of the orignal instrumented tensor as known to the debugger.
    """
    return self._tensor_aliases.get(tensor.name, tensor.name)