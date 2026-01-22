import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _safe_close(self, sess):
    """Closes a session without raising an exception.

    Just like sess.close() but ignores exceptions.

    Args:
      sess: A `Session`.
    """
    try:
        sess.close()
    except Exception:
        pass