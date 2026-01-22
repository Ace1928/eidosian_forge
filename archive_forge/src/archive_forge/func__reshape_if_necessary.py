import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _reshape_if_necessary(tensor, new_shape):
    """Like reshape(), but avoids creating a new tensor if possible."""
    new_shape = tuple((-1 if x is None else x for x in new_shape))
    cur_shape = tuple((x.value for x in tensor.shape.dims))
    if len(new_shape) == len(cur_shape) and all((not isinstance(d1, tensor_lib.Tensor) and (d0 == d1 or d1 == -1) for d0, d1 in zip(cur_shape, new_shape))):
        return tensor
    else:
        return array_ops.reshape(tensor, new_shape)