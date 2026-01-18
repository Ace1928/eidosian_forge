import typing
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_api(array_ops.where_v2)
def where_v2(condition: ragged_tensor.RaggedOrDense, x: typing.Optional[ragged_tensor.RaggedOrDense]=None, y: typing.Optional[ragged_tensor.RaggedOrDense]=None, name=None):
    """Return the elements where `condition` is `True`.

  : If both `x` and `y` are None: Retrieve indices of true elements.

    Returns the coordinates of true elements of `condition`. The coordinates
    are returned in a 2-D tensor with shape
    `[num_true_values, dim_size(condition)]`, where `result[i]` is the
    coordinates of the `i`th true value (in row-major order).

  : If both `x` and `y` are non-`None`: Multiplex between `x` and `y`.

    Choose an output shape  from the shapes of `condition`, `x`, and `y` that
    all three shapes are broadcastable to; and then use the broadcasted
    `condition` tensor as a mask that chooses whether the corredsponding element
    in the output should be taken from `x` (if `condition` is true) or `y` (if
    `condition` is false).

  >>> # Example: retrieve indices of true elements
  >>> tf.where(tf.ragged.constant([[True, False], [True]]))
  <tf.Tensor: shape=(2, 2), dtype=int64, numpy= array([[0, 0], [1, 0]])>

  >>> # Example: multiplex between `x` and `y`
  >>> tf.where(tf.ragged.constant([[True, False], [True, False, True]]),
  ...          tf.ragged.constant([['A', 'B'], ['C', 'D', 'E']]),
  ...          tf.ragged.constant([['a', 'b'], ['c', 'd', 'e']]))
  <tf.RaggedTensor [[b'A', b'b'], [b'C', b'd', b'E']]>

  Args:
    condition: A potentially ragged tensor of type `bool`
    x: A potentially ragged tensor (optional).
    y: A potentially ragged tensor (optional).  Must be specified if `x` is
      specified.  Must have the same rank and type as `x`.
    name: A name of the operation (optional).

  Returns:
    : If both `x` and `y` are `None`:
      A `Tensor` with shape `(num_true, rank(condition))`.
    : Otherwise:
      A potentially ragged tensor with the same type as `x` and `y`, and whose
      shape is broadcast-compatible with `x`, `y`, and `condition`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-`None`; or when
      `condition`, `x`, and `y` have incompatible shapes.
  """
    if (x is None) != (y is None):
        raise ValueError('x and y must be either both None or both non-None')
    with ops.name_scope('RaggedWhere', name, [condition, x, y]):
        condition = ragged_tensor.convert_to_tensor_or_ragged_tensor(condition, name='condition')
        if x is None:
            return _coordinate_where(condition)
        else:
            x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
            y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, name='y')
            condition, x, y = ragged_tensor.match_row_splits_dtypes(condition, x, y)
            return _elementwise_where_v2(condition, x, y)