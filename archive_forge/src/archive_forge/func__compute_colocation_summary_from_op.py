import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
def _compute_colocation_summary_from_op(op, prefix=''):
    """Fetch colocation file, line, and nesting and return a summary string."""
    return _compute_colocation_summary_from_dict(op.name, op._colocation_dict, prefix)