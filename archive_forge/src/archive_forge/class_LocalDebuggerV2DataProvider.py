import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
class LocalDebuggerV2DataProvider(provider.DataProvider):
    """A DataProvider implementation for tfdbg v2 data on local filesystem.

    In this implementation, `experiment_id` is assumed to be the path to the
    logdir that contains the DebugEvent file set.
    """

    def __init__(self, logdir):
        """Constructor of LocalDebuggerV2DataProvider.

        Args:
          logdir: Path to the directory from which the tfdbg v2 data will be
            loaded.
        """
        super().__init__()
        self._multiplexer = debug_data_multiplexer.DebuggerV2EventMultiplexer(logdir)

    def list_runs(self, ctx=None, *, experiment_id):
        """List runs available.

        Args:
          experiment_id: currently unused, because the backing
            DebuggerV2EventMultiplexer does not accommodate multiple experiments.

        Returns:
          Run names as a list of str.
        """
        return [provider.Run(run_id=run, run_name=run, start_time=self._get_first_event_timestamp(run)) for run in self._multiplexer.Runs()]

    def _get_first_event_timestamp(self, run_name):
        try:
            return self._multiplexer.FirstEventTimestamp(run_name)
        except ValueError as e:
            return None

    def list_scalars(self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None):
        del experiment_id, plugin_name, run_tag_filter
        raise TypeError("Debugger V2 DataProvider doesn't support scalars.")

    def read_scalars(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
        del experiment_id, plugin_name, downsample, run_tag_filter
        raise TypeError("Debugger V2 DataProvider doesn't support scalars.")

    def list_blob_sequences(self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None):
        del experiment_id, plugin_name, run_tag_filter
        raise NotImplementedError()

    def read_blob_sequences(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
        del experiment_id, downsample
        if plugin_name != PLUGIN_NAME:
            raise ValueError('Unsupported plugin_name: %s' % plugin_name)
        if run_tag_filter.runs is None:
            raise ValueError('run_tag_filter.runs is expected to be specified, but is not.')
        if run_tag_filter.tags is None:
            raise ValueError('run_tag_filter.tags is expected to be specified, but is not.')
        output = dict()
        existing_runs = self._multiplexer.Runs()
        for run in run_tag_filter.runs:
            if run not in existing_runs:
                continue
            output[run] = dict()
            for tag in run_tag_filter.tags:
                if tag.startswith((ALERTS_BLOB_TAG_PREFIX, EXECUTION_DIGESTS_BLOB_TAG_PREFIX, EXECUTION_DATA_BLOB_TAG_PREFIX, GRAPH_EXECUTION_DIGESTS_BLOB_TAG_PREFIX, GRAPH_EXECUTION_DATA_BLOB_TAG_PREFIX, GRAPH_INFO_BLOB_TAG_PREFIX, GRAPH_OP_INFO_BLOB_TAG_PREFIX, SOURCE_FILE_BLOB_TAG_PREFIX, STACK_FRAMES_BLOB_TAG_PREFIX)) or tag in (SOURCE_FILE_LIST_BLOB_TAG,):
                    output[run][tag] = [provider.BlobReference(blob_key='%s.%s' % (tag, run))]
        return output

    def read_blob(self, ctx=None, *, blob_key):
        if blob_key.startswith(ALERTS_BLOB_TAG_PREFIX):
            run, begin, end, alert_type = _parse_alerts_blob_key(blob_key)
            return json.dumps(self._multiplexer.Alerts(run, begin, end, alert_type_filter=alert_type))
        elif blob_key.startswith(EXECUTION_DIGESTS_BLOB_TAG_PREFIX):
            run, begin, end = _parse_execution_digest_blob_key(blob_key)
            return json.dumps(self._multiplexer.ExecutionDigests(run, begin, end))
        elif blob_key.startswith(EXECUTION_DATA_BLOB_TAG_PREFIX):
            run, begin, end = _parse_execution_data_blob_key(blob_key)
            return json.dumps(self._multiplexer.ExecutionData(run, begin, end))
        elif blob_key.startswith(GRAPH_EXECUTION_DIGESTS_BLOB_TAG_PREFIX):
            run, begin, end = _parse_graph_execution_digest_blob_key(blob_key)
            return json.dumps(self._multiplexer.GraphExecutionDigests(run, begin, end))
        elif blob_key.startswith(GRAPH_EXECUTION_DATA_BLOB_TAG_PREFIX):
            run, begin, end = _parse_graph_execution_data_blob_key(blob_key)
            return json.dumps(self._multiplexer.GraphExecutionData(run, begin, end))
        elif blob_key.startswith(GRAPH_INFO_BLOB_TAG_PREFIX):
            run, graph_id = _parse_graph_info_blob_key(blob_key)
            return json.dumps(self._multiplexer.GraphInfo(run, graph_id))
        elif blob_key.startswith(GRAPH_OP_INFO_BLOB_TAG_PREFIX):
            run, graph_id, op_name = _parse_graph_op_info_blob_key(blob_key)
            return json.dumps(self._multiplexer.GraphOpInfo(run, graph_id, op_name))
        elif blob_key.startswith(SOURCE_FILE_LIST_BLOB_TAG):
            run = _parse_source_file_list_blob_key(blob_key)
            return json.dumps(self._multiplexer.SourceFileList(run))
        elif blob_key.startswith(SOURCE_FILE_BLOB_TAG_PREFIX):
            run, index = _parse_source_file_blob_key(blob_key)
            return json.dumps(self._multiplexer.SourceLines(run, index))
        elif blob_key.startswith(STACK_FRAMES_BLOB_TAG_PREFIX):
            run, stack_frame_ids = _parse_stack_frames_blob_key(blob_key)
            return json.dumps(self._multiplexer.StackFrames(run, stack_frame_ids))
        else:
            raise ValueError('Unrecognized blob_key: %s' % blob_key)