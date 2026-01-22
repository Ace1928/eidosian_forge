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
def _process_graph_def(self, graph_def):
    for node_def in graph_def.node:
        if debug_graphs.is_debug_node(node_def.name) and node_def.attr['gated_grpc'].b:
            node_name, output_slot, _, debug_op = debug_graphs.parse_debug_node_name(node_def.name)
            self._gated_grpc_debug_watches.add(DebugWatch(node_name, output_slot, debug_op))