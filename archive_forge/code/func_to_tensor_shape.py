from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def to_tensor_shape(spec):
    """Returns a tf.TensorShape object that matches the shape specifications.

  If the InputSpec's shape or ndim is defined, this method will return a fully
  or partially-known shape. Otherwise, the returned TensorShape is None.

  Args:
    spec: an InputSpec object.

  Returns:
    a tf.TensorShape object
  """
    if spec.ndim is None and spec.shape is None:
        return tensor_shape.TensorShape(None)
    elif spec.shape is not None:
        return tensor_shape.TensorShape(spec.shape)
    else:
        shape = [None] * spec.ndim
        for a in spec.axes:
            shape[a] = spec.axes[a]
        return tensor_shape.TensorShape(shape)