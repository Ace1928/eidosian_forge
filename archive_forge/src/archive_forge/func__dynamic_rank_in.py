import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _dynamic_rank_in(actual_rank, given_ranks):
    if len(given_ranks) < 1:
        return ops.convert_to_tensor(False)
    result = math_ops.equal(given_ranks[0], actual_rank)
    for given_rank in given_ranks[1:]:
        result = math_ops.logical_or(result, math_ops.equal(given_rank, actual_rank))
    return result