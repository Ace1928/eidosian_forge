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
def _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, x):
    """Implements the Sturm sequence recurrence."""
    with ops.name_scope('sturm'):
        n = alpha.shape[0]
        zeros = array_ops.zeros(array_ops.shape(x), dtype=dtypes.int32)
        ones = array_ops.ones(array_ops.shape(x), dtype=dtypes.int32)

        def sturm_step0():
            q = alpha[0] - x
            count = array_ops.where(q < 0, ones, zeros)
            q = array_ops.where(math_ops.equal(alpha[0], x), alpha0_perturbation, q)
            return (q, count)

        def sturm_step(i, q, count):
            q = alpha[i] - beta_sq[i - 1] / q - x
            count = array_ops.where(q <= pivmin, count + 1, count)
            q = array_ops.where(q <= pivmin, math_ops.minimum(q, -pivmin), q)
            return (q, count)
        q, count = sturm_step0()
        blocksize = 16
        i = 1
        peel = (n - 1) % blocksize
        unroll_cnt = peel

        def unrolled_steps(start, q, count):
            for j in range(unroll_cnt):
                q, count = sturm_step(start + j, q, count)
            return (start + unroll_cnt, q, count)
        i, q, count = unrolled_steps(i, q, count)
        unroll_cnt = blocksize
        cond = lambda i, q, count: math_ops.less(i, n)
        _, _, count = while_loop.while_loop(cond, unrolled_steps, [i, q, count], back_prop=False)
        return count