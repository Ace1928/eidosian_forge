import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
def FlushExecutionFiles(self):
    """Flush the execution debug event files.

    Causes the current content of the cyclic buffers to be written to
    the .execution and .graph_execution_traces debug events files.
    Also clears those cyclic buffers.
    """
    _pywrap_debug_events_writer.FlushExecutionFiles(self._dump_root)