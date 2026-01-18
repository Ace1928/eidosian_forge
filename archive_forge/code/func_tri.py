import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.tri', v1=[])
@np_utils.np_doc('tri')
def tri(N, M=None, k=0, dtype=None):
    M = M if M is not None else N
    if dtype is not None:
        dtype = np_utils.result_type(dtype)
    else:
        dtype = np_utils.result_type(float)
    if k < 0:
        lower = -k - 1
        if lower > N:
            r = array_ops.zeros([N, M], dtype)
        else:
            o = array_ops.ones([N, M], dtype=dtypes.bool)
            r = math_ops.cast(math_ops.logical_not(array_ops.matrix_band_part(o, lower, -1)), dtype)
    else:
        o = array_ops.ones([N, M], dtype)
        if k > M:
            r = o
        else:
            r = array_ops.matrix_band_part(o, -1, k)
    return r