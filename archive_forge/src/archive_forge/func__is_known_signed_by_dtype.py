import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
def _is_known_signed_by_dtype(dt):
    """Helper returning True if dtype is known to be signed."""
    return {dtypes.float16: True, dtypes.float32: True, dtypes.float64: True, dtypes.int8: True, dtypes.int16: True, dtypes.int32: True, dtypes.int64: True}.get(dt.base_dtype, False)