import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _compute_error(grad):
    if isinstance(grad, tuple):
        grad = [grad]
    error = 0
    for j_t, j_n in grad:
        if j_t.size or j_n.size:
            error = np.maximum(error, np.fabs(j_t - j_n).max())
    return error