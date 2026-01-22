import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.MonitoredSession'])
class MonitoredSession(_MonitoredSession):
    """Session-like object that handles initialization, recovery and hooks.

  Example usage:

  ```python
  saver_hook = CheckpointSaverHook(...)
  summary_hook = SummarySaverHook(...)
  with MonitoredSession(session_creator=ChiefSessionCreator(...),
                        hooks=[saver_hook, summary_hook]) as sess:
    while not sess.should_stop():
      sess.run(train_op)
  ```

  Initialization: At creation time the monitored session does following things
  in given order:

  * calls `hook.begin()` for each given hook
  * finalizes the graph via `scaffold.finalize()`
  * create session
  * initializes the model via initialization ops provided by `Scaffold`
  * restores variables if a checkpoint exists
  * launches queue runners
  * calls `hook.after_create_session()`

  Run: When `run()` is called, the monitored session does following things:

  * calls `hook.before_run()`
  * calls TensorFlow `session.run()` with merged fetches and feed_dict
  * calls `hook.after_run()`
  * returns result of `session.run()` asked by user
  * if `AbortedError` or `UnavailableError` occurs, it recovers or
    reinitializes the session before executing the run() call again


  Exit: At the `close()`, the monitored session does following things in order:

  * calls `hook.end()`
  * closes the queue runners and the session
  * suppresses `OutOfRange` error which indicates that all inputs have been
    processed if the monitored_session is used as a context

  How to set `tf.compat.v1.Session` arguments:

  * In most cases you can set session arguments as follows:

  ```python
  MonitoredSession(
    session_creator=ChiefSessionCreator(master=..., config=...))
  ```

  * In distributed setting for a non-chief worker, you can use following:

  ```python
  MonitoredSession(
    session_creator=WorkerSessionCreator(master=..., config=...))
  ```

  See `MonitoredTrainingSession` for an example usage based on chief or worker.

  Note: This is not a `tf.compat.v1.Session`. For example, it cannot do
  following:

  * it cannot be set as default session.
  * it cannot be sent to saver.save.
  * it cannot be sent to tf.train.start_queue_runners.

  @compatibility(TF2)
  This API is not compatible with eager execution and `tf.function`. To migrate
  to TF2, rewrite the code to be compatible with eager execution. Check the
  [migration
  guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
  on replacing `Session.run` calls. In Keras, session hooks can be replaced by
  Callbacks e.g. [logging hook notebook](
  https://github.com/tensorflow/docs/blob/master/site/en/guide/migrate/logging_stop_hook.ipynb)
  For more details please read [Better
  performance with tf.function](https://www.tensorflow.org/guide/function).
  @end_compatibility

  Args:
    session_creator: A factory object to create session. Typically a
      `ChiefSessionCreator` which is the default one.
    hooks: An iterable of `SessionRunHook' objects.

  Returns:
    A MonitoredSession object.
  """

    def __init__(self, session_creator=None, hooks=None, stop_grace_period_secs=120):
        super(MonitoredSession, self).__init__(session_creator, hooks, should_recover=True, stop_grace_period_secs=stop_grace_period_secs)