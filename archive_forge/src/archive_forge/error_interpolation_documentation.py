import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
Interpolates an error message.

  The error message can contain tags of form `{{node_type node_name}}`
  which will be parsed to identify the tf.Graph and op. If the op contains
  traceback, the traceback will be attached to the error message.

  Args:
    message: A string to interpolate.
    graph: ops.Graph object containing all nodes referenced in the error
        message.

  Returns:
    The error message string with node definition traceback.
  