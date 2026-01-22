import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _get_trace_dir(self):
    found, trace_dir = self.get_flag_value(FLAG_NAME_TRACE_DIR)
    if found and trace_dir and self.use_test_undeclared_outputs_dir():
        raise ValueError('Cannot not use --%s and --%s at the same time' % (FLAG_NAME_TRACE_DIR, FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR))
    if self.use_test_undeclared_outputs_dir():
        trace_dir = self._env.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
    return trace_dir