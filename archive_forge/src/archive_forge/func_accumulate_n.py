import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.accumulate_n', v1=['math.accumulate_n', 'accumulate_n'])
@dispatch.add_dispatch_support
@deprecation.deprecated(None, 'Use `tf.math.add_n` Instead')
def accumulate_n(inputs, shape=None, tensor_dtype=None, name=None):
    """Returns the element-wise sum of a list of tensors.

  Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
  otherwise, these are inferred.

  For example:

  >>> a = tf.constant([[1, 2], [3, 4]])
  >>> b = tf.constant([[5, 0], [0, 6]])
  >>> tf.math.accumulate_n([a, b, a]).numpy()
  array([[ 7, 4],
         [ 6, 14]], dtype=int32)

  >>> # Explicitly pass shape and type
  >>> tf.math.accumulate_n(
  ...     [a, b, a], shape=[2, 2], tensor_dtype=tf.int32).numpy()
  array([[ 7,  4],
         [ 6, 14]], dtype=int32)

  Note: The input must be a list or tuple. This function does not handle
  `IndexedSlices`

  See Also:

  * `tf.reduce_sum(inputs, axis=0)` - This performe the same mathematical
    operation, but `tf.add_n` may be more efficient because it sums the
    tensors directly. `reduce_sum` on the other hand calls
    `tf.convert_to_tensor` on the list of tensors, unncessairly stacking them
    into a single tensor before summing.
  * `tf.add_n` - This is another python wrapper for the same Op. It has
    nearly identical functionality.

  Args:
    inputs: A list of `Tensor` objects, each with same shape and type.
    shape: Expected shape of elements of `inputs` (optional). Also controls the
      output shape of this op, which may affect type inference in other ops. A
      value of `None` means "infer the input shape from the shapes in `inputs`".
    tensor_dtype: Expected data type of `inputs` (optional). A value of `None`
      means "infer the input dtype from `inputs[0]`".
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as the elements of `inputs`.

  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  """

    def _input_error():
        return ValueError('inputs must be a list of at least one Tensor with the same dtype and shape')
    if not inputs or not isinstance(inputs, (list, tuple)):
        raise _input_error()
    inputs = indexed_slices.convert_n_to_tensor_or_indexed_slices(inputs)
    if not all((isinstance(x, tensor_lib.Tensor) for x in inputs)):
        raise _input_error()
    if not all((x.dtype == inputs[0].dtype for x in inputs)):
        raise _input_error()
    if shape is not None:
        shape = tensor_shape.as_shape(shape)
    else:
        shape = tensor_shape.unknown_shape()
    for input_tensor in inputs:
        if isinstance(input_tensor, tensor_lib.Tensor):
            shape = shape.merge_with(input_tensor.get_shape())
    if tensor_dtype is not None and tensor_dtype != inputs[0].dtype:
        raise TypeError(f'The `tensor_dtype` argument is {tensor_dtype}, but `input` is of type {inputs[0].dtype}. These must be equal. Try casting the input to the desired type.')
    if len(inputs) == 1 and name is None:
        return inputs[0]
    elif len(inputs) == 1 and name is not None:
        return array_ops.identity(inputs[0], name=name)
    return add_n(inputs, name=name)