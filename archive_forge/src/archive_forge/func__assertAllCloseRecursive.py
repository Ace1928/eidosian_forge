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
def _assertAllCloseRecursive(self, a, b, rtol=1e-06, atol=1e-06, path=None, msg=None):
    if ragged_tensor.is_ragged(a) or ragged_tensor.is_ragged(b):
        return self._assertRaggedClose(a, b, rtol, atol, msg)
    path = path or []
    path_str = '[' + ']['.join((str(p) for p in path)) + ']' if path else ''
    msg = msg if msg else ''
    if hasattr(a, '_asdict'):
        a = a._asdict()
    if hasattr(b, '_asdict'):
        b = b._asdict()
    a_is_dict = isinstance(a, collections_abc.Mapping)
    if a_is_dict != isinstance(b, collections_abc.Mapping):
        raise ValueError("Can't compare dict to non-dict, a%s vs b%s. %s" % (path_str, path_str, msg))
    if a_is_dict:
        self.assertItemsEqual(a.keys(), b.keys(), msg='mismatched keys: a%s has keys %s, but b%s has keys %s. %s' % (path_str, a.keys(), path_str, b.keys(), msg))
        for k in a:
            path.append(k)
            self._assertAllCloseRecursive(a[k], b[k], rtol=rtol, atol=atol, path=path, msg=msg)
            del path[-1]
    elif isinstance(a, (list, tuple)):
        try:
            a, b = self.evaluate_if_both_tensors(a, b)
            a_as_ndarray = self._GetNdArray(a)
            b_as_ndarray = self._GetNdArray(b)
            self._assertArrayLikeAllClose(a_as_ndarray, b_as_ndarray, rtol=rtol, atol=atol, msg='Mismatched value: a%s is different from b%s. %s' % (path_str, path_str, msg))
        except (ValueError, TypeError, NotImplementedError) as e:
            if len(a) != len(b):
                raise ValueError('Mismatched length: a%s has %d items, but b%s has %d items. %s' % (path_str, len(a), path_str, len(b), msg))
            for idx, (a_ele, b_ele) in enumerate(zip(a, b)):
                path.append(str(idx))
                self._assertAllCloseRecursive(a_ele, b_ele, rtol=rtol, atol=atol, path=path, msg=msg)
                del path[-1]
    else:
        try:
            self._assertArrayLikeAllClose(a, b, rtol=rtol, atol=atol, msg='Mismatched value: a%s is different from b%s. %s' % (path_str, path_str, msg))
        except TypeError as e:
            msg = 'Error: a%s has %s, but b%s has %s. %s' % (path_str, type(a), path_str, type(b), msg)
            e.args = (e.args[0] + ' : ' + msg,) + e.args[1:]
            raise