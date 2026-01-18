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
`merge_call()` implementation for synchronized replica.

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

    