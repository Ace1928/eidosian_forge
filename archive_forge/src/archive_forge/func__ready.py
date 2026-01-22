import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _ready(op, sess, msg):
    """Checks if the model is ready or not, as determined by op.

  Args:
    op: An op, either _ready_op or _ready_for_local_init_op, which defines the
      readiness of the model.
    sess: A `Session`.
    msg: A message to log to warning if not ready

  Returns:
    A tuple (is_ready, msg), where is_ready is True if ready and False
    otherwise, and msg is `None` if the model is ready, a `String` with the
    reason why it is not ready otherwise.
  """
    if op is None:
        return (True, None)
    else:
        try:
            ready_value = sess.run(op)
            if ready_value is None or ready_value.dtype == np.int32 or ready_value.size == 0:
                return (True, None)
            else:
                non_initialized_varnames = ', '.join([i.decode('utf-8') for i in ready_value])
                return (False, 'Variables not initialized: ' + non_initialized_varnames)
        except errors.FailedPreconditionError as e:
            if 'uninitialized' not in str(e):
                logging.warning('%s : error [%s]', msg, str(e))
                raise e
            return (False, str(e))