import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
class BaseMonitor(object):
    """Base class for debug event data monitors."""

    def __init__(self, debug_events_reader):
        self._debug_data_reader = debug_events_reader
        debug_events_reader._add_monitor(self)

    def on_execution(self, execution_index, execution):
        """Monitor method for top-level execution events.

    Return values (if any) are ignored by the associated DebugDataReader.

    Args:
      execution_index: The index of the top-level execution event, as an int.
      execution: An Execution data object, for a top-level op or function
        execution event.
    """

    def on_graph_execution_trace(self, graph_execution_trace_index, graph_execution_trace):
        """Monitor method for intra-graph execution events.

    Return values (if any) are ignored by the associated DebugDataReader.

    Args:
      graph_execution_trace_index: The index of the intra-graph execution
        event, as an int.
      graph_execution_trace: A GraphExecutionTrace data object, for an
        intra-graph tensor event.
    """