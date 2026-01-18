import numbers
import sys
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.tile', v1=[])
@np_utils.np_doc_only('tile')
def tile(a, reps):
    a = np_array_ops.array(a)
    reps = array_ops.reshape(np_array_ops.array(reps, dtype=dtypes.int32), [-1])
    a_rank = array_ops.rank(a)
    reps_size = array_ops.size(reps)
    reps = array_ops.pad(reps, [[math_ops.maximum(a_rank - reps_size, 0), 0]], constant_values=1)
    a_shape = array_ops.pad(array_ops.shape(a), [[math_ops.maximum(reps_size - a_rank, 0), 0]], constant_values=1)
    a = array_ops.reshape(a, a_shape)
    return array_ops.tile(a, reps)