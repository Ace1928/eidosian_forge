import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _load_execution(self):
    """Incrementally read the .execution file."""
    execution_iter = self._reader.execution_iterator()
    for debug_event, offset in execution_iter:
        self._execution_digests.append(_execution_digest_from_debug_event_proto(debug_event, offset))
        if self._monitors:
            execution = _execution_from_debug_event_proto(debug_event, offset)
            for monitor in self._monitors:
                monitor.on_execution(len(self._execution_digests) - 1, execution)