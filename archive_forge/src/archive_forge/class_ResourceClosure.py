import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class ResourceClosure(Closure):
    """A closure that builds a resource on a worker.

  ResourceClosures keep a reference to the closure object, which is used to
  rerun the closure upon recovery to ensure  workers have access to the
  resources they need.
  """

    def _init_remote_value(self):
        return RemoteValueImpl(self, self._output_type_spec)

    def build_output_remote_value(self):
        if self._output_remote_value_ref is None:
            ret = self._init_remote_value()
            self._output_remote_value_ref = weakref.ref(ret)
            return ret
        else:
            return self._output_remote_value_ref()