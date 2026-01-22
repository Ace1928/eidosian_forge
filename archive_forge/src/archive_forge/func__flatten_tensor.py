from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _flatten_tensor(tensor, sequence_mask, expected_length):
    """Flattens the two first dimensions and reshapes a tensor or sparse tensor.

  If `tensor` is a dense tensor, the sequence_mask is used to infer valid
  inputs.

  Note: If `tensor` is a `SparseTensor` and the indices are not sorted, they
  will be reordered.

  Args:
    tensor: A `Tensor` or `SparseTensor` of dimension at least 2, of shape
      [batch_size, seq_length, D0, D1, ..., DN].
    sequence_mask: A boolean `Tensor` of shape [batch_size, seq_length].
    expected_length: A integer scalar `Tensor` with the expected length of the
      resulting flattenned Tensor.

  Returns:
    A `Tensor` object of shape [expected_length, D0, D1, ..., DN].

  Raises:
    ValueError: If `tensor` has not at least 2 dimensions.
    ValueError: If `tensor` is not a `Tensor` or `SparseTensor` object.
    InvalidArgumentError: If the resulting `Tensor` doesn't have the expected
      length.
  """
    shape = tensor.get_shape()
    if shape.ndims < 2:
        raise ValueError('Input tensor expected to have at least 2 dimensions, got {} instead.'.format(shape.ndims))
    if isinstance(tensor, tf.sparse.SparseTensor):
        flat_tensor = tf.sparse.reorder(tensor).values
        if shape.ndims > 2:
            new_shape = tf.concat([[-1], shape[2:]], axis=0)
            flat_tensor = tf.reshape(tensor.values, new_shape)
    elif isinstance(tensor, tf.Tensor):
        flat_tensor = tf.boolean_mask(tensor, sequence_mask)
    else:
        raise ValueError('`tensor` expected to be a `Tensor` or  `SparseTensor` got `{}` instead.'.format(tensor))
    if shape.ndims == 2:
        flat_tensor = tf.compat.v1.expand_dims(flat_tensor, -1)
        expected_shape = tf.concat([[expected_length], [1]], axis=0)
    else:
        expected_shape = tf.concat([[expected_length], shape[2:]], axis=0)
    err_message = 'Tensor shape is incompatible with provided mask.'
    if tf.executing_eagerly():
        if flat_tensor._shape_tuple() != tuple(expected_shape.numpy()):
            raise ValueError(err_message)
        return flat_tensor
    with tf.control_dependencies([tf.compat.v1.debugging.assert_equal(tf.compat.v1.shape(flat_tensor), expected_shape, message=err_message)]):
        return tf.identity(flat_tensor)