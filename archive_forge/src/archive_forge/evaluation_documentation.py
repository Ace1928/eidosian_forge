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
Constructs the run hook.

    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      log_progress: Whether to log evaluation progress, defaults to True.
    