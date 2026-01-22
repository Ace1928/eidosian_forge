import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
def _EnsureTimestampAdded(self, debug_event):
    if debug_event.wall_time == 0:
        debug_event.wall_time = time.time()