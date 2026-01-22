from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _shapes(tensor_list_list, shapes, enqueue_many):
    """Calculate and merge the shapes of incoming tensors.

  Args:
    tensor_list_list: List of tensor lists.
    shapes: List of shape tuples corresponding to tensors within the lists.
    enqueue_many: Boolean describing whether shapes will be enqueued as
      batches or individual entries.

  Returns:
    A list of shapes aggregating shape inference info from `tensor_list_list`,
    or returning `shapes` if it is not `None`.

  Raises:
    ValueError: If any of the inferred shapes in `tensor_list_list` lack a
      well defined rank.
  """
    if shapes is None:
        len0 = len(tensor_list_list[0])
        for tl in tensor_list_list:
            for i in range(len0):
                if tl[i].shape.ndims is None:
                    raise ValueError("Cannot infer Tensor's rank: %s" % tl[i])
        shapes = [_merge_shapes([tl[i].shape.as_list() for tl in tensor_list_list], enqueue_many) for i in range(len0)]
    return shapes