import math
import numbers
import os
import re
import sys
import time
import types
from absl import app
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _get_name(self, overwrite_name=None):
    """Returns full name of class and method calling report_benchmark."""
    stack = tf_inspect.stack()
    calling_class = None
    name = None
    for frame in stack[::-1]:
        f_locals = frame[0].f_locals
        f_self = f_locals.get('self', None)
        if isinstance(f_self, Benchmark):
            calling_class = f_self
            name = frame[3]
            break
    if calling_class is None:
        raise ValueError('Unable to determine calling Benchmark class.')
    name = overwrite_name or name
    class_name = type(calling_class).__name__
    name = '%s.%s' % (class_name, name)
    return name