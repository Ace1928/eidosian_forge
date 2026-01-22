from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_FEATURE_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_LABEL_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_RECEIVER_DEFAULT_NAME
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _placeholder_from_tensor(t, default_batch_size=None):
    """Creates a placeholder that matches the dtype and shape of passed tensor.

  Args:
    t: Tensor or EagerTensor
    default_batch_size: the number of query examples expected per batch. Leave
      unset for variable batch size (recommended).

  Returns:
    Placeholder that matches the passed tensor.
  """
    batch_shape = tf.TensorShape([default_batch_size])
    shape = batch_shape.concatenate(t.get_shape()[1:])
    try:
        name = t.op.name
    except AttributeError:
        name = None
    return tf.compat.v1.placeholder(dtype=t.dtype, shape=shape, name=name)