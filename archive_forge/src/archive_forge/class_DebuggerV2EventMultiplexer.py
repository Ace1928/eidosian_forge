import threading
from tensorboard import errors
class DebuggerV2EventMultiplexer:
    """A class used for accessing tfdbg v2 DebugEvent data on local filesystem.

    This class is a short-term hack, mirroring the EventMultiplexer for the main
    TensorBoard plugins (e.g., scalar, histogram and graphs.) As such, it only
    implements the methods relevant to the Debugger V2 pluggin.

    TODO(cais): Integrate it with EventMultiplexer and use the integrated class
    from MultiplexerDataProvider for a single path of accessing debugger and
    non-debugger data.
    """

    def __init__(self, logdir):
        """Constructor for the `DebugEventMultiplexer`.

        Args:
          logdir: Path to the directory to load the tfdbg v2 data from.
        """
        self._logdir = logdir
        self._reader = None
        self._reader_lock = threading.Lock()
        self._reload_needed_event = None
        self._tryCreateReader()

    def _tryCreateReader(self):
        """Try creating reader for tfdbg2 data in the logdir.

        If the reader has already been created, a new one will not be created and
        this function is a no-op.

        If a reader has not been created, create it and start periodic calls to
        `update()` on a separate thread.
        """
        if self._reader:
            return
        with self._reader_lock:
            if not self._reader:
                try:
                    from tensorflow.python.debug.lib import debug_events_reader
                    from tensorflow.python.debug.lib import debug_events_monitors
                except ImportError:
                    return
                try:
                    self._reader = debug_events_reader.DebugDataReader(self._logdir)
                except AttributeError:
                    return
                except ValueError:
                    return
                self._monitors = [debug_events_monitors.InfNanMonitor(self._reader, limit=DEFAULT_PER_TYPE_ALERT_LIMIT)]
                self._reload_needed_event, _ = run_repeatedly_in_background(self._reader.update, DEFAULT_RELOAD_INTERVAL_SEC)

    def _reloadReader(self):
        """If a reader exists and has started period updating, unblock the update.

        The updates are performed periodically with a sleep interval between
        successive calls to the reader's update() method. Calling this method
        interrupts the sleep immediately if one is ongoing.
        """
        if self._reload_needed_event:
            self._reload_needed_event.set()

    def FirstEventTimestamp(self, run):
        """Return the timestamp of the first DebugEvent of the given run.

        This may perform I/O if no events have been loaded yet for the run.

        Args:
          run: A string name of the run for which the timestamp is retrieved.
            This currently must be hardcoded as `DEFAULT_DEBUGGER_RUN_NAME`,
            as each logdir contains at most one DebugEvent file set (i.e., a
            run of a tfdbg2-instrumented TensorFlow program.)

        Returns:
            The wall_time of the first event of the run, which will be in seconds
            since the epoch as a `float`.
        """
        if self._reader is None:
            raise ValueError('No tfdbg2 runs exists.')
        if run != DEFAULT_DEBUGGER_RUN_NAME:
            raise ValueError('Expected run name to be %s, but got %s' % (DEFAULT_DEBUGGER_RUN_NAME, run))
        return self._reader.starting_wall_time()

    def PluginRunToTagToContent(self, plugin_name):
        raise NotImplementedError('DebugDataMultiplexer.PluginRunToTagToContent() has not been implemented yet.')

    def Runs(self):
        """Return all the tfdbg2 run names in the logdir watched by this instance.

        The `Run()` method of this class is specialized for the tfdbg2-format
        DebugEvent files.

        As a side effect, this method unblocks the underlying reader's period
        reloading if a reader exists. This lets the reader update at a higher
        frequency than the default one with 30-second sleeping period between
        reloading when data is being queried actively from this instance.
        Note that this `Runs()` method is used by all other public data-access
        methods of this class (e.g., `ExecutionData()`, `GraphExecutionData()`).
        Hence calls to those methods will lead to accelerated data reloading of
        the reader.

        Returns:
          If tfdbg2-format data exists in the `logdir` of this object, returns:
              ```
              {runName: { "debugger-v2": [tag1, tag2, tag3] } }
              ```
              where `runName` is the hard-coded string `DEFAULT_DEBUGGER_RUN_NAME`
              string. This is related to the fact that tfdbg2 currently contains
              at most one DebugEvent file set per directory.
          If no tfdbg2-format data exists in the `logdir`, an empty `dict`.
        """
        self._tryCreateReader()
        if self._reader:
            self._reloadReader()
            return {DEFAULT_DEBUGGER_RUN_NAME: {'debugger-v2': []}}
        else:
            return {}

    def _checkBeginEndIndices(self, begin, end, total_count):
        if begin < 0:
            raise errors.InvalidArgumentError('Invalid begin index (%d)' % begin)
        if end > total_count:
            raise errors.InvalidArgumentError('end index (%d) out of bounds (%d)' % (end, total_count))
        if end >= 0 and end < begin:
            raise errors.InvalidArgumentError('end index (%d) is unexpectedly less than begin index (%d)' % (end, begin))
        if end < 0:
            end = total_count
        return end

    def Alerts(self, run, begin, end, alert_type_filter=None):
        """Get alerts from the debugged TensorFlow program.

        Args:
          run: The tfdbg2 run to get Alerts from.
          begin: Beginning alert index.
          end: Ending alert index.
          alert_type_filter: Optional filter string for alert type, used to
            restrict retrieved alerts data to a single type. If used,
            `begin` and `end` refer to the beginning and ending indices within
            the filtered alert type.
        """
        from tensorflow.python.debug.lib import debug_events_monitors
        runs = self.Runs()
        if run not in runs:
            return None
        alerts = []
        alerts_breakdown = dict()
        alerts_by_type = dict()
        for monitor in self._monitors:
            monitor_alerts = monitor.alerts()
            if not monitor_alerts:
                continue
            alerts.extend(monitor_alerts)
            if isinstance(monitor, debug_events_monitors.InfNanMonitor):
                alert_type = 'InfNanAlert'
            else:
                alert_type = '__MiscellaneousAlert__'
            alerts_breakdown[alert_type] = len(monitor_alerts)
            alerts_by_type[alert_type] = monitor_alerts
        num_alerts = len(alerts)
        if alert_type_filter is not None:
            if alert_type_filter not in alerts_breakdown:
                raise errors.InvalidArgumentError('Filtering of alerts failed: alert type %s does not exist' % alert_type_filter)
            alerts = alerts_by_type[alert_type_filter]
        end = self._checkBeginEndIndices(begin, end, len(alerts))
        return {'begin': begin, 'end': end, 'alert_type': alert_type_filter, 'num_alerts': num_alerts, 'alerts_breakdown': alerts_breakdown, 'per_type_alert_limit': DEFAULT_PER_TYPE_ALERT_LIMIT, 'alerts': [_alert_to_json(alert) for alert in alerts[begin:end]]}

    def ExecutionDigests(self, run, begin, end):
        """Get ExecutionDigests.

        Args:
          run: The tfdbg2 run to get `ExecutionDigest`s from.
          begin: Beginning execution index.
          end: Ending execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
        runs = self.Runs()
        if run not in runs:
            return None
        execution_digests = self._reader.executions(digest=True)
        end = self._checkBeginEndIndices(begin, end, len(execution_digests))
        return {'begin': begin, 'end': end, 'num_digests': len(execution_digests), 'execution_digests': [digest.to_json() for digest in execution_digests[begin:end]]}

    def ExecutionData(self, run, begin, end):
        """Get Execution data objects (Detailed, non-digest form).

        Args:
          run: The tfdbg2 run to get `ExecutionDigest`s from.
          begin: Beginning execution index.
          end: Ending execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
        runs = self.Runs()
        if run not in runs:
            return None
        execution_digests = self._reader.executions(digest=True)
        end = self._checkBeginEndIndices(begin, end, len(execution_digests))
        execution_digests = execution_digests[begin:end]
        executions = self._reader.executions(digest=False, begin=begin, end=end)
        return {'begin': begin, 'end': end, 'executions': [execution.to_json() for execution in executions]}

    def GraphExecutionDigests(self, run, begin, end, trace_id=None):
        """Get `GraphExecutionTraceDigest`s.

        Args:
          run: The tfdbg2 run to get `GraphExecutionTraceDigest`s from.
          begin: Beginning graph-execution index.
          end: Ending graph-execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
        runs = self.Runs()
        if run not in runs:
            return None
        if trace_id is not None:
            raise NotImplementedError('trace_id support for GraphExecutionTraceDigest is not implemented yet.')
        graph_exec_digests = self._reader.graph_execution_traces(digest=True)
        end = self._checkBeginEndIndices(begin, end, len(graph_exec_digests))
        return {'begin': begin, 'end': end, 'num_digests': len(graph_exec_digests), 'graph_execution_digests': [digest.to_json() for digest in graph_exec_digests[begin:end]]}

    def GraphExecutionData(self, run, begin, end, trace_id=None):
        """Get `GraphExecutionTrace`s.

        Args:
          run: The tfdbg2 run to get `GraphExecutionTrace`s from.
          begin: Beginning graph-execution index.
          end: Ending graph-execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
        runs = self.Runs()
        if run not in runs:
            return None
        if trace_id is not None:
            raise NotImplementedError('trace_id support for GraphExecutionTraceData is not implemented yet.')
        digests = self._reader.graph_execution_traces(digest=True)
        end = self._checkBeginEndIndices(begin, end, len(digests))
        graph_executions = self._reader.graph_execution_traces(digest=False, begin=begin, end=end)
        return {'begin': begin, 'end': end, 'graph_executions': [graph_exec.to_json() for graph_exec in graph_executions]}

    def GraphInfo(self, run, graph_id):
        """Get the information regarding a TensorFlow graph.

        Args:
          run: Name of the run.
          graph_id: Debugger-generated ID of the graph in question.
            This information is available in the return values
            of `GraphOpInfo`, `GraphExecution`, etc.

        Returns:
          A JSON-serializable object containing the information regarding
            the TensorFlow graph.

        Raises:
          NotFoundError if the graph_id is not known to the debugger.
        """
        runs = self.Runs()
        if run not in runs:
            return None
        try:
            graph = self._reader.graph_by_id(graph_id)
        except KeyError:
            raise errors.NotFoundError('There is no graph with ID "%s"' % graph_id)
        return graph.to_json()

    def GraphOpInfo(self, run, graph_id, op_name):
        """Get the information regarding a graph op's creation.

        Args:
          run: Name of the run.
          graph_id: Debugger-generated ID of the graph that contains
            the op in question. This ID is available from other methods
            of this class, e.g., the return value of `GraphExecutionDigests()`.
          op_name: Name of the op.

        Returns:
          A JSON-serializable object containing the information regarding
            the op's creation and its immediate inputs and consumers.

        Raises:
          NotFoundError if the graph_id or op_name does not exist.
        """
        runs = self.Runs()
        if run not in runs:
            return None
        try:
            graph = self._reader.graph_by_id(graph_id)
        except KeyError:
            raise errors.NotFoundError('There is no graph with ID "%s"' % graph_id)
        try:
            op_creation_digest = graph.get_op_creation_digest(op_name)
        except KeyError:
            raise errors.NotFoundError('There is no op named "%s" in graph with ID "%s"' % (op_name, graph_id))
        data_object = self._opCreationDigestToDataObject(op_creation_digest, graph)
        for input_spec in data_object['inputs']:
            try:
                input_op_digest = graph.get_op_creation_digest(input_spec['op_name'])
            except KeyError:
                input_op_digest = None
            if input_op_digest:
                input_spec['data'] = self._opCreationDigestToDataObject(input_op_digest, graph)
        for slot_consumer_specs in data_object['consumers']:
            for consumer_spec in slot_consumer_specs:
                try:
                    digest = graph.get_op_creation_digest(consumer_spec['op_name'])
                except KeyError:
                    digest = None
                if digest:
                    consumer_spec['data'] = self._opCreationDigestToDataObject(digest, graph)
        return data_object

    def _opCreationDigestToDataObject(self, op_creation_digest, graph):
        if op_creation_digest is None:
            return None
        json_object = op_creation_digest.to_json()
        del json_object['graph_id']
        json_object['graph_ids'] = self._getGraphStackIds(op_creation_digest.graph_id)
        json_object['num_outputs'] = op_creation_digest.num_outputs
        del json_object['input_names']
        json_object['inputs'] = []
        for input_tensor_name in op_creation_digest.input_names or []:
            input_op_name, output_slot = parse_tensor_name(input_tensor_name)
            json_object['inputs'].append({'op_name': input_op_name, 'output_slot': output_slot})
        json_object['consumers'] = []
        for _ in range(json_object['num_outputs']):
            json_object['consumers'].append([])
        for src_slot, consumer_op_name, dst_slot in graph.get_op_consumers(json_object['op_name']):
            json_object['consumers'][src_slot].append({'op_name': consumer_op_name, 'input_slot': dst_slot})
        return json_object

    def _getGraphStackIds(self, graph_id):
        """Retrieve the IDs of all outer graphs of a graph.

        Args:
          graph_id: Id of the graph being queried with respect to its outer
            graphs context.

        Returns:
          A list of graph_ids, ordered from outermost to innermost, including
            the input `graph_id` argument as the last item.
        """
        graph_ids = [graph_id]
        graph = self._reader.graph_by_id(graph_id)
        while graph.outer_graph_id:
            graph_ids.insert(0, graph.outer_graph_id)
            graph = self._reader.graph_by_id(graph.outer_graph_id)
        return graph_ids

    def SourceFileList(self, run):
        runs = self.Runs()
        if run not in runs:
            return None
        return self._reader.source_file_list()

    def SourceLines(self, run, index):
        runs = self.Runs()
        if run not in runs:
            return None
        try:
            host_name, file_path = self._reader.source_file_list()[index]
        except IndexError:
            raise errors.NotFoundError('There is no source-code file at index %d' % index)
        return {'host_name': host_name, 'file_path': file_path, 'lines': self._reader.source_lines(host_name, file_path)}

    def StackFrames(self, run, stack_frame_ids):
        runs = self.Runs()
        if run not in runs:
            return None
        stack_frames = []
        for stack_frame_id in stack_frame_ids:
            if stack_frame_id not in self._reader._stack_frame_by_id:
                raise errors.NotFoundError('Cannot find stack frame with ID %s' % stack_frame_id)
            stack_frames.append(self._reader._stack_frame_by_id[stack_frame_id])
        return {'stack_frames': stack_frames}