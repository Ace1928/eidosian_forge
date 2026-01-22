import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _validate_flag_names(self):
    """Validates if the TensorTrace flags passed are valid."""
    tensor_tracer_flags = self._env.get(FLAGS_ENV_VAR)
    if not tensor_tracer_flags:
        return
    pos = 0
    while True:
        match, _ = TTParameters.match_next_flag(tensor_tracer_flags, pos)
        if not match:
            break
        flag_name = match.group(1)
        if flag_name not in VALID_FLAG_NAMES:
            raise ValueError('The flag name "%s" passed via the environment variable "%s" is invalid. Valid flag names are:\n%s' % (flag_name, FLAGS_ENV_VAR, VALID_FLAG_NAMES))
        pos = match.end()