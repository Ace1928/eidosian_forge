import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
def RegisterDeviceAndGetId(self, device_name):
    return _pywrap_debug_events_writer.RegisterDeviceAndGetId(self._dump_root, device_name)