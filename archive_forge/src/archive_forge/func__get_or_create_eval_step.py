import math
import time
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
def _get_or_create_eval_step():
    """Gets or creates the eval step `Tensor`.

  Returns:
    A `Tensor` representing a counter for the evaluation step.

  Raises:
    ValueError: If multiple `Tensors` have been added to the
      `tf.GraphKeys.EVAL_STEP` collection.
  """
    graph = ops.get_default_graph()
    eval_steps = graph.get_collection(ops.GraphKeys.EVAL_STEP)
    if len(eval_steps) == 1:
        return eval_steps[0]
    elif len(eval_steps) > 1:
        raise ValueError('Multiple tensors added to tf.GraphKeys.EVAL_STEP')
    else:
        counter = variable_scope.get_variable('eval_step', shape=[], dtype=dtypes.int64, initializer=init_ops.zeros_initializer(), trainable=False, collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.EVAL_STEP])
        return counter