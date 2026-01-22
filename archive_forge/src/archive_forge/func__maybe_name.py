import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _maybe_name(obj):
    """Returns object name if it has one, or a message otherwise.

  This is useful for names that apper in error messages.
  Args:
    obj: Object to get the name of.
  Returns:
    name, "None", or a "no name" message.
  """
    if obj is None:
        return 'None'
    elif hasattr(obj, 'name'):
        return obj.name
    else:
        return '<no name for %s>' % type(obj)