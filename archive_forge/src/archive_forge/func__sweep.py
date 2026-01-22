import collections
import os.path
import re
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _sweep(self):
    """Deletes or preserves managed checkpoints."""
    if not self._max_to_keep:
        return
    while len(self._maybe_delete) > self._max_to_keep:
        filename, timestamp = self._maybe_delete.popitem(last=False)
        if self._keep_checkpoint_every_n_hours and timestamp - self._keep_checkpoint_every_n_hours * 3600.0 >= self._last_preserved_timestamp:
            self._last_preserved_timestamp = timestamp
            continue
        _delete_file_if_exists(filename + '.index')
        _delete_file_if_exists(filename + '.data-?????-of-?????')