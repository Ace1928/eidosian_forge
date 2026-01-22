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
def _get_latest_eval_step_value(update_ops):
    """Gets the eval step `Tensor` value after running `update_ops`.

  Args:
    update_ops: A list of `Tensors` or a dictionary of names to `Tensors`, which
      are run before reading the eval step value.

  Returns:
    A `Tensor` representing the value for the evaluation step.
  """
    if isinstance(update_ops, dict):
        update_ops = list(update_ops.values())
    with ops.control_dependencies(update_ops):
        return array_ops.identity(_get_or_create_eval_step().read_value())