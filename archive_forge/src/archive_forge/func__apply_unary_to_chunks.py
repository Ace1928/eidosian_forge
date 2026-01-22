import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _apply_unary_to_chunks(f, chunks_by_dev):
    """Apply a unary op to each tensor in chunks_by_dev, on same device.

  Args:
    f: a unary function over `tf.Tensor`.
    chunks_by_dev: list of lists of `tf.Tensor`.

  Returns:
    new list of lists of `tf.Tensor` with the same structure as
    chunks_by_dev containing the derived tensors.
  """
    output = []
    for x in chunks_by_dev:
        with ops.colocate_with(x[0]):
            output.append([f(t) for t in x])
    return output