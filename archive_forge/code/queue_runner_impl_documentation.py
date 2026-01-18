import threading
import weakref
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Returns a `QueueRunner` object created from `queue_runner_def`.