import contextlib
import threading
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class DispatchContext(object):
    """Context entered when executing a closure on a given worker."""

    def __init__(self, worker_obj):
        self._worker = worker_obj
        self._worker_index = worker_obj.worker_index

    @property
    def worker(self):
        return self._worker

    @property
    def worker_index(self):
        return self._worker_index

    def maybe_get_remote_value(self, ret):
        return maybe_get_remote_value(ret)