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
def NCHW_VECT_CToNHWC(input_shape_or_tensor):
    """Transforms the input from the NCHW_VECT_C layout to NHWC layout.

  Note: Does not include de-quantization or type conversion steps, which should
  be applied beforehand.

  Args:
    input_shape_or_tensor: a 5- or 6-D tensor, or an array representing shape

  Returns:
    tensor or shape array transformed into NHWC

  Raises:
    ValueError: if last dimension of `input_shape_or_tensor` is not 4.
  """
    permutations = {5: [0, 2, 3, 1, 4], 6: [0, 2, 3, 4, 1, 5]}
    is_tensor = isinstance(input_shape_or_tensor, tensor_lib.Tensor)
    input_shape = input_shape_or_tensor.shape.as_list() if is_tensor else input_shape_or_tensor
    if input_shape[-1] != 4:
        raise ValueError('Last dimension of NCHW_VECT_C must be 4.')
    permutation = permutations[len(input_shape)]
    nhwc_shape = [input_shape[a] for a in permutation[:-1]]
    nhwc_shape[-1] *= input_shape[-1]
    if is_tensor:
        t = array_ops.transpose(input_shape_or_tensor, permutation)
        return array_ops.reshape(t, nhwc_shape)
    else:
        return nhwc_shape