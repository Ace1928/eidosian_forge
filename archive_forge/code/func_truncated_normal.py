import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
def truncated_normal(self, shape, mean, stddev, dtype):
    """A deterministic truncated normal if seed is passed."""
    if self.seed:
        op = stateless_random_ops.stateless_truncated_normal
    else:
        op = random_ops.truncated_normal
    return op(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)