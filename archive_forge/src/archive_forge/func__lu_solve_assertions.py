import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _lu_solve_assertions(lower_upper, perm, rhs, validate_args):
    """Returns list of assertions related to `lu_solve` assumptions."""
    assertions = lu_reconstruct_assertions(lower_upper, perm, validate_args)
    message = 'Input `rhs` must have at least 2 dimensions.'
    if rhs.shape.ndims is not None:
        if rhs.shape.ndims < 2:
            raise ValueError(message)
    elif validate_args:
        assertions.append(check_ops.assert_rank_at_least(rhs, rank=2, message=message))
    message = '`lower_upper.shape[-1]` must equal `rhs.shape[-1]`.'
    if lower_upper.shape[-1] is not None and rhs.shape[-2] is not None:
        if lower_upper.shape[-1] != rhs.shape[-2]:
            raise ValueError(message)
    elif validate_args:
        assertions.append(check_ops.assert_equal(array_ops.shape(lower_upper)[-1], array_ops.shape(rhs)[-2], message=message))
    return assertions