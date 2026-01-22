import contextlib
import threading
import weakref
from tensorflow.python import pywrap_tfe
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.util import traceback_utils
def _merge_call(self, fn, args, kwargs):
    """`merge_call()` implementation for synchronized replica.

    This pauses the current replica thread and passes `fn` and its arguments to
    the main thread. The main thread will wait until all replicas pause, then
    invoke `fn` with grouped arguments. The current replica thread will continue
    after `fn` completes.

    See `_call_for_each_replica` for the logic in the main thread.

    Args:
      fn: a function that is called in cross replica context with grouped
        arguments from each replica. `fn` should returns grouped values.
      args: positional arguments to `fn`.
      kwargs: keyward arguments to `fn`.

    Returns:
      Return value of `fn` for the current replica.

    Raises:
      RuntimeError: when merge_call happens in a different graph, e.g. in a
        different tf.function, which is not supported now.
      _RequestedStop: when stop is requested.

    """
    t = threading.current_thread()
    assert isinstance(t, _MirroredReplicaThread)
    t.merge_fn = fn
    t.merge_args = args
    t.merge_kwargs = kwargs
    t.captured_name_scope = t.graph.get_name_scope()
    if t.captured_name_scope:
        t.captured_name_scope += '/'
    t.captured_var_scope = variable_scope.get_variable_scope()
    t.captured_control_deps = t.graph._current_control_dependencies()
    t.merge_call_entered_in_eager = context.context().executing_eagerly()
    if ops.get_default_graph() != t.graph:
        raise RuntimeError('`merge_call` called while defining a new graph or a tf.function. This can often happen if the function `fn` passed to `strategy.run()` contains a nested `@tf.function`, and the nested `@tf.function` contains a synchronization point, such as aggregating gradients (e.g, optimizer.apply_gradients), or if the function `fn` uses a control flow statement which contains a synchronization point in the body. Such behaviors are not yet supported. Instead, please avoid nested `tf.function`s or control flow statements that may potentially cross a synchronization boundary, for example, wrap the `fn` passed to `strategy.run` or the entire `strategy.run` inside a `tf.function` or move the control flow out of `fn`. If you are subclassing a `tf.keras.Model`, please avoid decorating overridden methods `test_step` and `train_step` in `tf.function`.')
    t.has_paused.set()
    t.should_run.wait()
    t.should_run.clear()
    if t.coord.should_stop():
        raise _RequestedStop()
    t.merge_call_entered_in_eager = None
    return t.merge_result