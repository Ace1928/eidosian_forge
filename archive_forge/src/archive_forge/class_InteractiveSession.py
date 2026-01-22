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
@tf_export(v1=['InteractiveSession'])
class InteractiveSession(BaseSession):
    """A TensorFlow `Session` for use in interactive contexts, such as a shell.

  The only difference with a regular `Session` is that an `InteractiveSession`
  installs itself as the default session on construction.
  The methods `tf.Tensor.eval`
  and `tf.Operation.run`
  will use that session to run ops.

  This is convenient in interactive shells and [IPython
  notebooks](http://ipython.org), as it avoids having to pass an explicit
  `Session` object to run ops.

  For example:

  ```python
  sess = tf.compat.v1.InteractiveSession()
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b
  # We can just use 'c.eval()' without passing 'sess'
  print(c.eval())
  sess.close()
  ```

  Note that a regular session installs itself as the default session when it
  is created in a `with` statement.  The common usage in non-interactive
  programs is to follow that pattern:

  ```python
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b
  with tf.compat.v1.Session():
    # We can also use 'c.eval()' here.
    print(c.eval())
  ```
  """
    _count_lock = threading.Lock()
    _active_session_count = 0

    def __init__(self, target='', graph=None, config=None):
        """Creates a new interactive TensorFlow session.

    If no `graph` argument is specified when constructing the session,
    the default graph will be launched in the session. If you are
    using more than one graph (created with `tf.Graph()`) in the same
    process, you will have to use different sessions for each graph,
    but each graph can be used in multiple sessions. In this case, it
    is often clearer to pass the graph to be launched explicitly to
    the session constructor.

    Args:
      target: (Optional.) The execution engine to connect to. Defaults to using
        an in-process engine.
      graph: (Optional.) The `Graph` to be launched (described above).
      config: (Optional) `ConfigProto` proto used to configure the session.
    """
        if not config:
            gpu_options = config_pb2.GPUOptions(allow_growth=True)
            config = config_pb2.ConfigProto(gpu_options=gpu_options)
        config.graph_options.place_pruned_graph = True
        super(InteractiveSession, self).__init__(target, graph, config)
        with InteractiveSession._count_lock:
            if InteractiveSession._active_session_count > 0:
                warnings.warn('An interactive session is already active. This can cause out-of-memory errors in some cases. You must explicitly call `InteractiveSession.close()` to release resources held by the other session(s).')
            InteractiveSession._active_session_count += 1
        self._explicitly_closed = False
        self._default_session = self.as_default()
        self._default_session.enforce_nesting = False
        self._default_session.__enter__()
        self._explicit_graph = graph
        if self._explicit_graph is not None:
            self._default_graph = graph.as_default()
            self._default_graph.enforce_nesting = False
            self._default_graph.__enter__()

    def close(self):
        """Closes an `InteractiveSession`."""
        super(InteractiveSession, self).close()
        with InteractiveSession._count_lock:
            if not self._explicitly_closed:
                InteractiveSession._active_session_count -= 1
                self._explicitly_closed = True
            else:
                return
        if self._explicit_graph is not None:
            self._default_graph.__exit__(None, None, None)
            self._default_graph = None
        self._default_session.__exit__(None, None, None)
        self._default_session = None