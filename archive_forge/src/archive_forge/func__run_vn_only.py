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
def _run_vn_only(func=None, v2=True, reason=None):
    """Execute the decorated test only if running in the specified mode.

  This function is intended to be applied to tests that exercise functionality
   that belongs to either only v2, or v1.
   If the test is run in the mode opposite of the specified one, it will simply
   be skipped.

   It shouldn't be used directly, instead, use the `run_v1_only` or
   `run_v2_only` wrappers that call it.

  `deprecated_graph_mode_only`, `run_v1_only`, `run_v2_only`, and
  `run_in_graph_and_eager_modes` are available decorators for different
  v1/v2/eager/graph combinations.

  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.
    v2: a boolean value indicating whether the test should be skipped in v2,
      or v1.
    reason: string giving a reason for limiting the test to a particular mode.

  Returns:
    Returns a decorator that will conditionally skip the decorated test method.
  """
    if not reason:
        reason = f'Test is only compatible with {('v2 ' if v2 else 'v1')}'

    def decorator(f):
        if tf_inspect.isclass(f):
            for cls in type.mro(f):
                setup = cls.__dict__.get('setUp')
                if setup is not None:
                    setattr(f, 'setUp', decorator(setup))
                    break
            return f
        else:

            def decorated(self, *args, **kwargs):
                tf2_enabled = tf2.enabled()
                if tf2_enabled and (not v2) or (not tf2_enabled and v2):
                    self.skipTest(reason)
                return f(self, *args, **kwargs)
            return decorated
    if func is not None:
        return decorator(func)
    return decorator