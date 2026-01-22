import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _get_report_filepath(self):
    """Sets the path of the output report file."""
    found, report_file_path = self.get_flag_value(FLAG_NAME_REPORT_FILE)
    if found and report_file_path and self.use_test_undeclared_outputs_dir():
        if os.path.isabs(report_file_path):
            raise ValueError('If use_test_undeclared_outputs_dir is set,report_file_path cannot be an absolute path (%s)' % report_file_path)
        outputs_dir = self._env.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
        report_file_path = os.path.join(outputs_dir, report_file_path)
    return report_file_path