import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _get_flag_int_value(self, wanted_flag_name, default_value):
    """Returns the int value of a TensorTracer flag.

    Args:
      wanted_flag_name: the name of the flag we are looking for.
      default_value: the default value for the flag, if not provided.
    Returns:
      the value of the flag.
    Raises:
      RuntimeError: If supposedly deadcode is reached.
    """
    flag_int_value = default_value
    found, flag_value = self.get_flag_value(wanted_flag_name)
    if found:
        try:
            flag_int_value = int(flag_value)
        except ValueError:
            logging.warning('Cannot convert %s to int for flag %s' % (flag_int_value, wanted_flag_name))
    return flag_int_value