import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
def _SparseMatMul(t1, t2, out_dtype, transpose_a=False, transpose_b=False):
    """Helper function to create SparseMatMul op."""
    assert t1.ref() in is_sparse and t2.ref() in is_sparse
    t1_sparse = is_sparse[t1.ref()]
    t2_sparse = is_sparse[t2.ref()]
    if transpose_b:
        t2 = array_ops.transpose(t2)
        transpose_b = False
    prod = math_ops.matmul(t1, t2, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=t1_sparse, b_is_sparse=t2_sparse)
    if prod.dtype != out_dtype:
        prod = math_ops.cast(prod, out_dtype)
    return prod