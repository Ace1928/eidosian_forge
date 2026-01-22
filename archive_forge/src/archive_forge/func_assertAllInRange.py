import collections
from collections import OrderedDict
import contextlib
import functools
import gc
import itertools
import math
import os
import random
import re
import tempfile
import threading
import time
import unittest
from absl.testing import parameterized
import numpy as np
from google.protobuf import descriptor_pool
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.client import session
from tensorflow.python.compat.compat import forward_compatibility_horizon
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _test_metrics_util
from tensorflow.python.framework import config
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import gen_sync_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_ops  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import _pywrap_stacktrace_handler
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import _pywrap_util_port
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.tf_export import tf_export
@py_func_if_in_function
def assertAllInRange(self, target, lower_bound, upper_bound, open_lower_bound=False, open_upper_bound=False):
    """Assert that elements in a Tensor are all in a given range.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      lower_bound: lower bound of the range
      upper_bound: upper bound of the range
      open_lower_bound: (`bool`) whether the lower bound is open (i.e., > rather
        than the default >=)
      open_upper_bound: (`bool`) whether the upper bound is open (i.e., < rather
        than the default <=)

    Raises:
      AssertionError:
        if the value tensor does not have an ordered numeric type (float* or
          int*), or
        if there are nan values, or
        if any of the elements do not fall in the specified range.
    """
    target = self._GetNdArray(target)
    if not (np.issubdtype(target.dtype, np.floating) or np.issubdtype(target.dtype, np.integer)):
        raise AssertionError('The value of %s does not have an ordered numeric type, instead it has type: %s' % (target, target.dtype))
    nan_subscripts = np.where(np.isnan(target))
    if np.size(nan_subscripts):
        raise AssertionError('%d of the %d element(s) are NaN. Subscripts(s) and value(s) of the NaN element(s):\n' % (len(nan_subscripts[0]), np.size(target)) + '\n'.join(self._format_subscripts(nan_subscripts, target)))
    range_str = ('(' if open_lower_bound else '[') + str(lower_bound) + ', ' + str(upper_bound) + (')' if open_upper_bound else ']')
    violations = np.less_equal(target, lower_bound) if open_lower_bound else np.less(target, lower_bound)
    violations = np.logical_or(violations, np.greater_equal(target, upper_bound) if open_upper_bound else np.greater(target, upper_bound))
    violation_subscripts = np.where(violations)
    if np.size(violation_subscripts):
        raise AssertionError('%d of the %d element(s) are outside the range %s. ' % (len(violation_subscripts[0]), np.size(target), range_str) + 'Subscript(s) and value(s) of the offending elements:\n' + '\n'.join(self._format_subscripts(violation_subscripts, target)))