import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
class DebugDataReader:
    """A reader that reads structured debugging data in the tfdbg v2 format.

  The set of data read by an object of this class concerns the execution history
  of a tfdbg2-instrumented TensorFlow program.

  Note:
    - An object of this class incrementally reads data from files that belong to
      the tfdbg v2 DebugEvent file set. Calling `update()` triggers the reading
      from the last-successful reading positions in the files.
    - This object can be used as a context manager. Its `__exit__()` call
      closes the file readers cleanly.
  """

    def __init__(self, dump_root):
        self._reader = DebugEventsReader(dump_root)
        self._execution_digests = []
        self._host_name_file_path_to_offset = collections.OrderedDict()
        self._stack_frame_by_id = dict()
        self._unprocessed_stack_frames = dict()
        self._device_by_id = dict()
        self._graph_by_id = dict()
        self._graph_op_digests = []
        self._graph_execution_trace_digests = []
        self._monitors = []

    def _add_monitor(self, monitor):
        self._monitors.append(monitor)

    def _load_source_files(self):
        """Incrementally read the .source_files DebugEvent file."""
        source_files_iter = self._reader.source_files_iterator()
        for debug_event, offset in source_files_iter:
            source_file = debug_event.source_file
            self._host_name_file_path_to_offset[source_file.host_name, source_file.file_path] = offset

    def _load_stack_frames(self):
        """Incrementally read the .stack_frames file.

    This must be called after _load_source_files().
    It assumes that the following contract is honored by the writer of the tfdbg
    v2 data file set:
      - Before a stack frame is written to the .stack_frames file, the
        corresponding source file information must have been written to the
        .source_files file first.
    """
        stack_frames_iter = self._reader.stack_frames_iterator()
        for debug_event, _ in stack_frames_iter:
            stack_frame_with_id = debug_event.stack_frame_with_id
            file_line_col = stack_frame_with_id.file_line_col
            self._unprocessed_stack_frames[stack_frame_with_id.id] = file_line_col
        unprocessed_stack_frame_ids = tuple(self._unprocessed_stack_frames.keys())
        for stack_frame_id in unprocessed_stack_frame_ids:
            file_line_col = self._unprocessed_stack_frames[stack_frame_id]
            if len(self._host_name_file_path_to_offset) > file_line_col.file_index:
                host_name, file_path = list(self._host_name_file_path_to_offset.keys())[file_line_col.file_index]
                self._stack_frame_by_id[stack_frame_id] = (host_name, file_path, file_line_col.line, file_line_col.func)
            del self._unprocessed_stack_frames[stack_frame_id]

    def _load_graphs(self):
        """Incrementally read the .graphs file.

    Compiles the DebuggedGraph and GraphOpCreation data.
    """
        graphs_iter = self._reader.graphs_iterator()
        for debug_event, offset in graphs_iter:
            if debug_event.graph_op_creation.ByteSize():
                op_creation_proto = debug_event.graph_op_creation
                op_digest = GraphOpCreationDigest(debug_event.wall_time, offset, op_creation_proto.graph_id, op_creation_proto.op_type, op_creation_proto.op_name, tuple(op_creation_proto.output_tensor_ids), op_creation_proto.code_location.host_name, tuple(op_creation_proto.code_location.stack_frame_ids), input_names=tuple(op_creation_proto.input_names))
                self._graph_op_digests.append(op_digest)
                debugged_graph = self._graph_by_id[op_creation_proto.graph_id]
                debugged_graph.add_op(op_digest)
                for dst_slot, input_name in enumerate(op_creation_proto.input_names):
                    src_op_name, src_slot = input_name.split(':')
                    debugged_graph.add_op_consumer(src_op_name, int(src_slot), op_creation_proto.op_name, dst_slot)
            elif debug_event.debugged_graph.ByteSize():
                graph_proto = debug_event.debugged_graph
                graph = DebuggedGraph(graph_proto.graph_name or None, graph_proto.graph_id, outer_graph_id=graph_proto.outer_context_id or None)
                self._graph_by_id[graph_proto.graph_id] = graph
                if graph_proto.outer_context_id:
                    self._graph_by_id[graph_proto.outer_context_id].add_inner_graph_id(graph.graph_id)
            elif debug_event.debugged_device.ByteSize():
                device_proto = debug_event.debugged_device
                self._device_by_id[device_proto.device_id] = DebuggedDevice(device_proto.device_name, device_proto.device_id)

    def _load_graph_execution_traces(self):
        """Incrementally load the .graph_execution_traces file."""
        for i, traces_iter in enumerate(self._reader.graph_execution_traces_iterators()):
            for debug_event, offset in traces_iter:
                self._graph_execution_trace_digests.append(self._graph_execution_trace_digest_from_debug_event_proto(debug_event, (i, offset)))
                if self._monitors:
                    graph_execution_trace = self._graph_execution_trace_from_debug_event_proto(debug_event, (i, offset))
                    for monitor in self._monitors:
                        monitor.on_graph_execution_trace(len(self._graph_execution_trace_digests) - 1, graph_execution_trace)

    def _graph_execution_trace_digest_from_debug_event_proto(self, debug_event, locator):
        trace_proto = debug_event.graph_execution_trace
        op_name = trace_proto.op_name
        op_type = self._lookup_op_type(trace_proto.tfdbg_context_id, op_name)
        return GraphExecutionTraceDigest(debug_event.wall_time, locator, op_type, op_name, trace_proto.output_slot, debug_event.graph_execution_trace.tfdbg_context_id)

    def _graph_execution_trace_from_debug_event_proto(self, debug_event, locator):
        """Convert a DebugEvent proto into a GraphExecutionTrace data object."""
        trace_proto = debug_event.graph_execution_trace
        graph_ids = [trace_proto.tfdbg_context_id]
        while True:
            graph = self.graph_by_id(graph_ids[0])
            if graph.outer_graph_id:
                graph_ids.insert(0, graph.outer_graph_id)
            else:
                break
        if trace_proto.tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
            debug_tensor_value = None
        else:
            debug_tensor_value = _parse_tensor_value(trace_proto.tensor_proto, return_list=True)
        return GraphExecutionTrace(self._graph_execution_trace_digest_from_debug_event_proto(debug_event, locator), graph_ids=graph_ids, tensor_debug_mode=trace_proto.tensor_debug_mode, debug_tensor_value=debug_tensor_value, device_name=trace_proto.device_name or None)

    def _lookup_op_type(self, graph_id, op_name):
        """Lookup the type of an op by name and the immediately enclosing graph.

    Args:
      graph_id: Debugger-generated ID of the immediately-enclosing graph.
      op_name: Name of the op.

    Returns:
      Op type as a str.
    """
        return self._graph_by_id[graph_id].get_op_creation_digest(op_name).op_type

    def _load_execution(self):
        """Incrementally read the .execution file."""
        execution_iter = self._reader.execution_iterator()
        for debug_event, offset in execution_iter:
            self._execution_digests.append(_execution_digest_from_debug_event_proto(debug_event, offset))
            if self._monitors:
                execution = _execution_from_debug_event_proto(debug_event, offset)
                for monitor in self._monitors:
                    monitor.on_execution(len(self._execution_digests) - 1, execution)

    def update(self):
        """Perform incremental read of the file set."""
        self._load_source_files()
        self._load_stack_frames()
        self._load_graphs()
        self._load_graph_execution_traces()
        self._load_execution()

    def source_file_list(self):
        """Get a list of source files known to the debugger data reader.

    Returns:
      A tuple of `(host_name, file_path)` tuples.
    """
        return tuple(self._host_name_file_path_to_offset.keys())

    def source_lines(self, host_name, file_path):
        """Read the line-by-line content of a source file.

    Args:
      host_name: Host name on which the source file is located.
      file_path: File path at which the source file is located.

    Returns:
      Lines of the source file as a `list` of `str`s.
    """
        offset = self._host_name_file_path_to_offset[host_name, file_path]
        return list(self._reader.read_source_files_event(offset).source_file.lines)

    def starting_wall_time(self):
        """Wall timestamp for when the debugged TensorFlow program started.

    Returns:
      Stating wall time as seconds since the epoch, as a `float`.
    """
        return self._reader.starting_wall_time()

    def tensorflow_version(self):
        """TensorFlow version used in the debugged TensorFlow program.

    Note: this is not necessarily the same as the version of TensorFlow used to
    load the DebugEvent file set.

    Returns:
      TensorFlow version used by the debugged program, as a `str`.
    """
        return self._reader.tensorflow_version()

    def tfdbg_run_id(self):
        """Get the debugger run ID of the debugged TensorFlow program."""
        return self._reader.tfdbg_run_id()

    def outermost_graphs(self):
        """Get the number of outer most graphs read so far."""
        return [graph for graph in self._graph_by_id.values() if not graph.outer_graph_id]

    def graph_by_id(self, graph_id):
        """Get a DebuggedGraph object by its ID."""
        return self._graph_by_id[graph_id]

    def device_name_by_id(self, device_id):
        """Get the name of a device by the debugger-generated ID of the device."""
        return self._device_by_id[device_id].device_name

    def device_name_map(self):
        """Get a map mapping device IDs to device names."""
        return {device_id: self._device_by_id[device_id].device_name for device_id in self._device_by_id}

    def graph_op_digests(self, op_type=None):
        """Get the list of the digests for graph-op creation so far.

    Args:
      op_type: Optional op type to filter the creation events with.

    Returns:
      A list of `GraphOpCreationDigest` objects.
    """
        if op_type is not None:
            return [digest for digest in self._graph_op_digests if digest.op_type == op_type]
        else:
            return self._graph_op_digests

    def graph_execution_traces(self, digest=False, begin=None, end=None):
        """Get all the intra-graph execution tensor traces read so far.

    Args:
      digest: Whether the results will be returned in the more light-weight
        digest form.
      begin: Optional beginning index for the requested traces or their digests.
        Python-style negative indices are supported.
      end: Optional ending index for the requested traces or their digests.
        Python-style negative indices are supported.

    Returns:
      If `digest`: a `list` of `GraphExecutionTraceDigest` objects.
      Else: a `list` of `GraphExecutionTrace` objects.
    """
        digests = self._graph_execution_trace_digests
        if begin is not None or end is not None:
            begin = begin or 0
            end = end or len(digests)
            digests = digests[begin:end]
        if digest:
            return digests
        else:
            return [self.read_graph_execution_trace(digest) for digest in digests]

    def num_graph_execution_traces(self):
        """Get the number of graph execution traces read so far."""
        return len(self._graph_execution_trace_digests)

    def executions(self, digest=False, begin=None, end=None):
        """Get `Execution`s or `ExecutionDigest`s this reader has read so far.

    Args:
      digest: Whether the results are returned in a digest form, i.e.,
        `ExecutionDigest` format, instead of the more detailed `Execution`
        format.
      begin: Optional beginning index for the requested execution data objects
        or their digests. Python-style negative indices are supported.
      end: Optional ending index for the requested execution data objects or
        their digests. Python-style negative indices are supported.

    Returns:
      If `digest`: a `list` of `ExecutionDigest` objects.
      Else: a `list` of `Execution` objects.
    """
        digests = self._execution_digests
        if begin is not None or end is not None:
            begin = begin or 0
            end = end or len(digests)
            digests = digests[begin:end]
        if digest:
            return digests
        else:
            return [self.read_execution(digest) for digest in digests]

    def num_executions(self):
        """Get the number of execution events read so far."""
        return len(self._execution_digests)

    def read_execution(self, execution_digest):
        """Read a detailed Execution object."""
        debug_event = self._reader.read_execution_event(execution_digest.locator)
        return _execution_from_debug_event_proto(debug_event, execution_digest.locator)

    def read_graph_execution_trace(self, graph_execution_trace_digest):
        """Read the detailed graph execution trace.

    Args:
      graph_execution_trace_digest: A `GraphExecutionTraceDigest` object.

    Returns:
      The corresponding `GraphExecutionTrace` object.
    """
        debug_event = self._reader.read_graph_execution_traces_event(graph_execution_trace_digest.locator)
        return self._graph_execution_trace_from_debug_event_proto(debug_event, graph_execution_trace_digest.locator)

    def read_execution_stack_trace(self, execution):
        """Read the stack trace of a given Execution object.

    Args:
      execution: The Execution object of interest.

    Returns:
      1. The host name.
      2. The stack trace, as a list of (file_path, lineno, func) tuples.
    """
        host_name = self._stack_frame_by_id[execution.stack_frame_ids[0]][0]
        return (host_name, [self._stack_frame_by_id[frame_id][1:] for frame_id in execution.stack_frame_ids])

    def read_graph_op_creation_stack_trace(self, graph_op_creation_digest):
        """Read the stack trace of a given graph op creation object.

    Args:
      graph_op_creation_digest: The GraphOpCreationDigest object of interest.

    Returns:
      A tuple consisting of:
        1. The host name.
        2. The stack trace, as a list of (file_path, lineno, func) tuples.
    """
        return (graph_op_creation_digest.host_name, [self._stack_frame_by_id[frame_id][1:] for frame_id in graph_op_creation_digest.stack_frame_ids])

    def execution_to_tensor_values(self, execution):
        """Read the full tensor values from an Execution or ExecutionDigest.

    Args:
      execution: An `ExecutionDigest` or `ExeuctionDigest` object.

    Returns:
      A list of numpy arrays representing the output tensor values of the
        execution event.
    """
        debug_event = self._reader.read_execution_event(execution.locator)
        return [_parse_tensor_value(tensor_proto) for tensor_proto in debug_event.execution.tensor_protos]

    def graph_execution_trace_to_tensor_value(self, trace):
        """Read full tensor values from an Execution or ExecutionDigest.

    Args:
      trace: An `GraphExecutionTraceDigest` or `GraphExecutionTrace` object.

    Returns:
      A numpy array representing the output tensor value of the intra-graph
        tensor execution event.
    """
        debug_event = self._reader.read_graph_execution_traces_event(trace.locator)
        return _parse_tensor_value(debug_event.graph_execution_trace.tensor_proto)

    def symbolic_tensor_id(self, graph_id, op_name, output_slot):
        """Get the ID of a symbolic tensor.

    Args:
      graph_id: The ID of the immediately-enclosing graph.
      op_name: Name of the op.
      output_slot: Output slot as an int.

    Returns:
      The ID of the symbolic tensor as an int.
    """
        return self._graph_by_id[graph_id].get_tensor_id(op_name, output_slot)

    def graph_execution_trace_to_tensor_id(self, trace):
        """Get symbolic tensor ID from a GraphExecutoinTraceDigest object."""
        return self.symbolic_tensor_id(trace.graph_id, trace.op_name, trace.output_slot)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        del exception_type, exception_value, traceback
        self._reader.close()