import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['Session'])
class Session(BaseSession):
    """A class for running TensorFlow operations.

  A `Session` object encapsulates the environment in which `Operation`
  objects are executed, and `Tensor` objects are evaluated. For
  example:

  ```python
  tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
  # Build a graph.
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b

  # Launch the graph in a session.
  sess = tf.compat.v1.Session()

  # Evaluate the tensor `c`.
  print(sess.run(c)) # prints 30.0
  ```

  A session may own resources, such as
  `tf.Variable`, `tf.queue.QueueBase`,
  and `tf.compat.v1.ReaderBase`. It is important to release
  these resources when they are no longer required. To do this, either
  invoke the `tf.Session.close` method on the session, or use
  the session as a context manager. The following two examples are
  equivalent:

  ```python
  # Using the `close()` method.
  sess = tf.compat.v1.Session()
  sess.run(...)
  sess.close()

  # Using the context manager.
  with tf.compat.v1.Session() as sess:
    sess.run(...)
  ```

  The
  [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
  protocol buffer exposes various configuration options for a
  session. For example, to create a session that uses soft constraints
  for device placement, and log the resulting placement decisions,
  create a session as follows:

  ```python
  # Launch the graph in a session that allows soft device placement and
  # logs the placement decisions.
  sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True))
  ```

  @compatibility(TF2)
  `Session` does not work with either eager execution or `tf.function`, and you
  should not invoke it directly. To migrate code that uses sessions to TF2,
  rewrite the code without it. See the
  [migration
  guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
  on replacing `Session.run` calls.
  @end_compatibility
  """

    def __init__(self, target='', graph=None, config=None):
        """Creates a new TensorFlow session.

    If no `graph` argument is specified when constructing the session,
    the default graph will be launched in the session. If you are
    using more than one graph (created with `tf.Graph()`) in the same
    process, you will have to use different sessions for each graph,
    but each graph can be used in multiple sessions. In this case, it
    is often clearer to pass the graph to be launched explicitly to
    the session constructor.

    Args:
      target: (Optional.) The execution engine to connect to. Defaults to using
        an in-process engine. See
        [Distributed TensorFlow](https://tensorflow.org/deploy/distributed) for
          more examples.
      graph: (Optional.) The `Graph` to be launched (described above).
      config: (Optional.) A
        [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
          protocol buffer with configuration options for the session.
    """
        super(Session, self).__init__(target, graph, config=config)
        self._default_graph_context_manager = None
        self._default_session_context_manager = None

    def __enter__(self):
        if self._default_graph_context_manager is None:
            self._default_graph_context_manager = self.graph.as_default()
        else:
            raise RuntimeError('Session context managers are not re-entrant. Use `Session.as_default()` if you want to enter a session multiple times.')
        if self._default_session_context_manager is None:
            self._default_session_context_manager = self.as_default()
        self._default_graph_context_manager.__enter__()
        return self._default_session_context_manager.__enter__()

    def __exit__(self, exec_type, exec_value, exec_tb):
        if exec_type is errors.OpError:
            logging.error('Session closing due to OpError: %s', (exec_value,))
        try:
            self._default_session_context_manager.__exit__(exec_type, exec_value, exec_tb)
        except RuntimeError as error:
            if error == exec_value:
                pass
            else:
                raise
        self._default_graph_context_manager.__exit__(exec_type, exec_value, exec_tb)
        self._default_session_context_manager = None
        self._default_graph_context_manager = None
        if exec_type:
            close_thread = threading.Thread(name='SessionCloseThread', target=self.close)
            close_thread.daemon = True
            close_thread.start()
            close_thread.join(30.0)
            if close_thread.is_alive():
                logging.error('Session failed to close after 30 seconds. Continuing after this point may leave your program in an undefined state.')
        else:
            self.close()

    @staticmethod
    def reset(target, containers=None, config=None):
        """Resets resource containers on `target`, and close all connected sessions.

    A resource container is distributed across all workers in the
    same cluster as `target`.  When a resource container on `target`
    is reset, resources associated with that container will be cleared.
    In particular, all Variables in the container will become undefined:
    they lose their values and shapes.

    NOTE:
    (i) reset() is currently only implemented for distributed sessions.
    (ii) Any sessions on the master named by `target` will be closed.

    If no resource containers are provided, all containers are reset.

    Args:
      target: The execution engine to connect to.
      containers: A list of resource container name strings, or `None` if all of
        all the containers are to be reset.
      config: (Optional.) Protocol buffer with configuration options.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        resetting containers.
    """
        if target is not None:
            target = compat.as_bytes(target)
        if containers is not None:
            containers = [compat.as_bytes(c) for c in containers]
        else:
            containers = []
        tf_session.TF_Reset(target, containers, config)