import collections
import threading
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util.tf_export import tf_export
@tf_export('profiler.experimental.Profile', v1=[])
class Profile(object):
    """Context-manager profile API.

  Profiling will start when entering the scope, and stop and save the results to
  the logdir when exits the scope. Open TensorBoard profile tab to view results.

  Example usage:
  ```python
  with tf.profiler.experimental.Profile("/path/to/logdir"):
    # do some work
  ```
  """

    def __init__(self, logdir, options=None):
        """Creates a context manager object for profiler API.

    Args:
      logdir: profile data will save to this directory.
      options: An optional `tf.profiler.experimental.ProfilerOptions` can be
        provided to fine tune the profiler's behavior.
    """
        self._logdir = logdir
        self._options = options

    def __enter__(self):
        start(self._logdir, self._options)

    def __exit__(self, typ, value, tb):
        stop()