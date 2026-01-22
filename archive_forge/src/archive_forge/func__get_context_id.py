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
def _get_context_id(self, context):
    """Get a unique ID for an op-construction context (e.g., a graph).

    If the graph has been encountered before, reuse the same unique ID.
    When encountering a new context (graph), this methods writes a DebugEvent
    proto with the debugged_graph field to the proper DebugEvent file.

    Args:
      context: A context to get the unique ID for. Must be hashable. E.g., a
        Graph object.

    Returns:
      A unique ID for the context.
    """
    if context in self._context_to_id:
        return self._context_to_id[context]
    graph_is_new = False
    with self._context_lock:
        if context not in self._context_to_id:
            graph_is_new = True
            context_id = _get_id()
            self._context_to_id[context] = context_id
    if graph_is_new:
        self.get_writer().WriteDebuggedGraph(debug_event_pb2.DebuggedGraph(graph_id=context_id, graph_name=getattr(context, 'name', None), outer_context_id=self._get_outer_context_id(context)))
    return self._context_to_id[context]