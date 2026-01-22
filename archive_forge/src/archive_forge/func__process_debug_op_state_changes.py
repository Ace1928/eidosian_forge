import collections
import json
import queue
import threading
import time
from concurrent import futures
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _process_debug_op_state_changes(self, event_reply=None):
    """Dequeue and process all the queued debug-op state change protos.

    Include all the debug-op state change protos in a `EventReply` proto.

    Args:
      event_reply: An `EventReply` to add the `DebugOpStateChange` protos to,
        or `None`.

    Returns:
      An `EventReply` proto with the dequeued `DebugOpStateChange` protos (if
        any) added.
    """
    if event_reply is None:
        event_reply = debug_service_pb2.EventReply()
    while not self._debug_ops_state_change_queue.empty():
        state_change = self._debug_ops_state_change_queue.get()
        debug_node_key = (state_change.node_name, state_change.output_slot, state_change.debug_op)
        if state_change.state == debug_service_pb2.EventReply.DebugOpStateChange.READ_WRITE:
            logging.info('Adding breakpoint %s:%d:%s', state_change.node_name, state_change.output_slot, state_change.debug_op)
            self._breakpoints.add(debug_node_key)
        elif state_change.state == debug_service_pb2.EventReply.DebugOpStateChange.READ_ONLY:
            logging.info('Adding watchpoint %s:%d:%s', state_change.node_name, state_change.output_slot, state_change.debug_op)
            if debug_node_key in self._breakpoints:
                self._breakpoints.discard(debug_node_key)
        elif state_change.state == debug_service_pb2.EventReply.DebugOpStateChange.DISABLED:
            logging.info('Removing watchpoint or breakpoint: %s:%d:%s', state_change.node_name, state_change.output_slot, state_change.debug_op)
            if debug_node_key in self._breakpoints:
                self._breakpoints.discard(debug_node_key)
            else:
                logging.warn('Attempting to remove a non-existent debug node key: %s', debug_node_key)
        new_state_change = event_reply.debug_op_state_changes.add()
        new_state_change.CopyFrom(state_change)
    return event_reply