import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _flag_value_to_re_list(self, flag_name):
    """Converts list of strings to compiled RE."""
    re_list = []
    found, flag_value = self.get_flag_value(flag_name)
    if not found or not flag_value:
        return re_list
    list_of_values = flag_value.split(',')
    for v in list_of_values:
        r = re.compile(v)
        re_list.append(r)
    return re_list