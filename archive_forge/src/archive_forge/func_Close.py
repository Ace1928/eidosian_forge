import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
def Close(self):
    """Close the writer."""
    _pywrap_debug_events_writer.Close(self._dump_root)