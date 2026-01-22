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
@tf_export(v1=['train.queue_runner.add_queue_runner', 'train.add_queue_runner'])
@deprecation.deprecated(None, _DEPRECATION_INSTRUCTION)
def add_queue_runner(qr, collection=ops.GraphKeys.QUEUE_RUNNERS):
    """Adds a `QueueRunner` to a collection in the graph.

  When building a complex model that uses many queues it is often difficult to
  gather all the queue runners that need to be run.  This convenience function
  allows you to add a queue runner to a well known collection in the graph.

  The companion method `start_queue_runners()` can be used to start threads for
  all the collected queue runners.

  @compatibility(TF2)
  QueueRunners are not compatible with eager execution. Instead, please
  use [tf.data](https://www.tensorflow.org/guide/data) to get data into your
  model.
  @end_compatibility

  Args:
    qr: A `QueueRunner`.
    collection: A `GraphKey` specifying the graph collection to add
      the queue runner to.  Defaults to `GraphKeys.QUEUE_RUNNERS`.
  """
    ops.add_to_collection(collection, qr)