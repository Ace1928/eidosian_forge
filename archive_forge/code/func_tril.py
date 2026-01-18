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
@tf_export.tf_export('experimental.numpy.tril', v1=[])
@np_utils.np_doc('tril')
def tril(m, k=0):
    m = asarray(m)
    if m.shape.ndims is None:
        raise ValueError('Argument to tril should have known rank')
    m_shape = m.shape.as_list()
    if len(m_shape) < 2:
        raise ValueError('Argument to tril must have rank at least 2')
    if m_shape[-1] is None or m_shape[-2] is None:
        raise ValueError('Currently, the last two dimensions of the input array need to be known.')
    z = constant_op.constant(0, m.dtype)
    mask = tri(*m_shape[-2:], k=k, dtype=bool)
    return array_ops.where_v2(array_ops.broadcast_to(mask, array_ops.shape(m)), m, z)