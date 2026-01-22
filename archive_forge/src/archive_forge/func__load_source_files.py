import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _load_source_files(self):
    """Incrementally read the .source_files DebugEvent file."""
    source_files_iter = self._reader.source_files_iterator()
    for debug_event, offset in source_files_iter:
        source_file = debug_event.source_file
        self._host_name_file_path_to_offset[source_file.host_name, source_file.file_path] = offset