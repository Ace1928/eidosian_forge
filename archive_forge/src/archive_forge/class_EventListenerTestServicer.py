import collections
import errno
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import portpicker
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.util import compat
class EventListenerTestServicer(grpc_debug_server.EventListenerBaseServicer):
    """An implementation of EventListenerBaseServicer for testing."""

    def __init__(self, server_port, dump_dir, toggle_watch_on_core_metadata=None):
        """Constructor of EventListenerTestServicer.

    Args:
      server_port: (int) The server port number.
      dump_dir: (str) The root directory to which the data files will be
        dumped. If empty or None, the received debug data will not be dumped
        to the file system: they will be stored in memory instead.
      toggle_watch_on_core_metadata: A list of
        (node_name, output_slot, debug_op) tuples to toggle the
        watchpoint status during the on_core_metadata calls (optional).
    """
        self.core_metadata_json_strings = []
        self.partition_graph_defs = []
        self.debug_tensor_values = collections.defaultdict(list)
        self._initialize_toggle_watch_state(toggle_watch_on_core_metadata)
        grpc_debug_server.EventListenerBaseServicer.__init__(self, server_port, functools.partial(EventListenerTestStreamHandler, dump_dir, self))
        self._call_types = []
        self._call_keys = []
        self._origin_stacks = []
        self._origin_id_to_strings = []
        self._graph_tracebacks = []
        self._graph_versions = []
        self._source_files = []

    def _initialize_toggle_watch_state(self, toggle_watches):
        self._toggle_watches = toggle_watches
        self._toggle_watch_state = {}
        if self._toggle_watches:
            for watch_key in self._toggle_watches:
                self._toggle_watch_state[watch_key] = False

    def toggle_watch(self):
        for watch_key in self._toggle_watch_state:
            node_name, output_slot, debug_op = watch_key
            if self._toggle_watch_state[watch_key]:
                self.request_unwatch(node_name, output_slot, debug_op)
            else:
                self.request_watch(node_name, output_slot, debug_op)
            self._toggle_watch_state[watch_key] = not self._toggle_watch_state[watch_key]

    def clear_data(self):
        self.core_metadata_json_strings = []
        self.partition_graph_defs = []
        self.debug_tensor_values = collections.defaultdict(list)
        self._call_types = []
        self._call_keys = []
        self._origin_stacks = []
        self._origin_id_to_strings = []
        self._graph_tracebacks = []
        self._graph_versions = []
        self._source_files = []

    def SendTracebacks(self, request, context):
        self._call_types.append(request.call_type)
        self._call_keys.append(request.call_key)
        self._origin_stacks.append(request.origin_stack)
        self._origin_id_to_strings.append(request.origin_id_to_string)
        self._graph_tracebacks.append(request.graph_traceback)
        self._graph_versions.append(request.graph_version)
        return debug_service_pb2.EventReply()

    def SendSourceFiles(self, request, context):
        self._source_files.append(request)
        return debug_service_pb2.EventReply()

    def query_op_traceback(self, op_name):
        """Query the traceback of an op.

    Args:
      op_name: Name of the op to query.

    Returns:
      The traceback of the op, as a list of 3-tuples:
        (filename, lineno, function_name)

    Raises:
      ValueError: If the op cannot be found in the tracebacks received by the
        server so far.
    """
        for op_log_proto in self._graph_tracebacks:
            for log_entry in op_log_proto.log_entries:
                if log_entry.name == op_name:
                    return self._code_def_to_traceback(log_entry.code_def, op_log_proto.id_to_string)
        raise ValueError("Op '%s' does not exist in the tracebacks received by the debug server." % op_name)

    def query_origin_stack(self):
        """Query the stack of the origin of the execution call.

    Returns:
      A `list` of all tracebacks. Each item corresponds to an execution call,
        i.e., a `SendTracebacks` request. Each item is a `list` of 3-tuples:
        (filename, lineno, function_name).
    """
        ret = []
        for stack, id_to_string in zip(self._origin_stacks, self._origin_id_to_strings):
            ret.append(self._code_def_to_traceback(stack, id_to_string))
        return ret

    def query_call_types(self):
        return self._call_types

    def query_call_keys(self):
        return self._call_keys

    def query_graph_versions(self):
        return self._graph_versions

    def query_source_file_line(self, file_path, lineno):
        """Query the content of a given line in a source file.

    Args:
      file_path: Path to the source file.
      lineno: Line number as an `int`.

    Returns:
      Content of the line as a string.

    Raises:
      ValueError: If no source file is found at the given file_path.
    """
        if not self._source_files:
            raise ValueError('This debug server has not received any source file contents yet.')
        for source_files in self._source_files:
            for source_file_proto in source_files.source_files:
                if source_file_proto.file_path == file_path:
                    return source_file_proto.lines[lineno - 1]
        raise ValueError('Source file at path %s has not been received by the debug server', file_path)

    def _code_def_to_traceback(self, code_def, id_to_string):
        return [(id_to_string[trace.file_id], trace.lineno, id_to_string[trace.function_id]) for trace in code_def.traces]