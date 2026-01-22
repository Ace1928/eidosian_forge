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
def _maybe_validate_matrix(a, validate_args):
    """Checks that input is a `float` matrix."""
    assertions = []
    if not a.dtype.is_floating:
        raise TypeError('Input `a` must have `float`-like `dtype` (saw {}).'.format(a.dtype.name))
    if a.shape is not None and a.shape.rank is not None:
        if a.shape.rank < 2:
            raise ValueError('Input `a` must have at least 2 dimensions (saw: {}).'.format(a.shape.rank))
    elif validate_args:
        assertions.append(check_ops.assert_rank_at_least(a, rank=2, message='Input `a` must have at least 2 dimensions.'))
    return assertions