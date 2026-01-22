from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('SparseDenseCwiseMul')
def _SparseDenseCwiseMulGrad(op, grad):
    """Gradients for SparseDenseCwiseMul."""
    return _SparseDenseCwiseMulOrDivGrad(op, grad, True)