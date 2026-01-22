import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
def WriteGraphOpCreation(self, graph_op_creation):
    """Write a GraphOpCreation proto with the writer.

    Args:
      graph_op_creation: A GraphOpCreation proto, describing the details of the
        creation of an op inside a TensorFlow Graph.
    """
    debug_event = debug_event_pb2.DebugEvent(graph_op_creation=graph_op_creation)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteGraphOpCreation(self._dump_root, debug_event)