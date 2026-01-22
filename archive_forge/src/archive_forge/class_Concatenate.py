from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
class Concatenate(_Merge):
    """Layer that concatenates a list of inputs.

  It takes as input a list of tensors, all of the same shape except
  for the concatenation axis, and returns a single tensor that is the
  concatenation of all inputs.

  >>> x = np.arange(20).reshape(2, 2, 5)
  >>> print(x)
  [[[ 0  1  2  3  4]
    [ 5  6  7  8  9]]
   [[10 11 12 13 14]
    [15 16 17 18 19]]]
  >>> y = np.arange(20, 30).reshape(2, 1, 5)
  >>> print(y)
  [[[20 21 22 23 24]]
   [[25 26 27 28 29]]]
  >>> tf.keras.layers.Concatenate(axis=1)([x, y])
  <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
  array([[[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [20, 21, 22, 23, 24]],
         [[10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [25, 26, 27, 28, 29]]])>

  >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
  >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
  >>> concatted = tf.keras.layers.Concatenate()([x1, x2])
  >>> concatted.shape
  TensorShape([5, 16])

  """

    def __init__(self, axis=-1, **kwargs):
        """Instantiates a Concatenate layer.

    >>> x = np.arange(20).reshape(2, 2, 5)
    >>> print(x)
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
     [[10 11 12 13 14]
      [15 16 17 18 19]]]
    >>> y = np.arange(20, 30).reshape(2, 1, 5)
    >>> print(y)
    [[[20 21 22 23 24]]
     [[25 26 27 28 29]]]
    >>> tf.keras.layers.Concatenate(axis=1)([x, y])
    <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [20, 21, 22, 23, 24]],
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [25, 26, 27, 28, 29]]])>

    Args:
      axis: Axis along which to concatenate.
      **kwargs: standard layer keyword arguments.
    """
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if not isinstance(input_shape[0], tuple) or len(input_shape) < 1:
            raise ValueError('A `Concatenate` layer should be called on a list of at least 1 input.')
        if all((shape is None for shape in input_shape)):
            return
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) != 1:
            err_msg = 'A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got inputs shapes: %s' % input_shape
            ranks = set((len(shape) for shape in shape_set))
            if len(ranks) != 1:
                raise ValueError(err_msg)
            rank, = ranks
            for axis in range(rank):
                unique_dims = set((shape[axis] for shape in shape_set if shape[axis] is not None))
                if len(unique_dims) > 1:
                    raise ValueError(err_msg)

    def _merge_function(self, inputs):
        return backend.concatenate(inputs, axis=self.axis)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or not isinstance(input_shape[0], (tuple, list)):
            raise ValueError('A `Concatenate` layer should be called on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, (tuple, list)):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, (tuple, list)):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` should have the same length.')
        if all((m is None for m in mask)):
            return None
        masks = []
        for input_i, mask_i in zip(inputs, mask):
            if mask_i is None:
                masks.append(array_ops.ones_like(input_i, dtype='bool'))
            elif backend.ndim(mask_i) < backend.ndim(input_i):
                masks.append(array_ops.expand_dims(mask_i, axis=-1))
            else:
                masks.append(mask_i)
        concatenated = backend.concatenate(masks, axis=self.axis)
        return backend.all(concatenated, axis=-1, keepdims=False)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Concatenate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))