from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import six
import tensorflow as tf
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_constants
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _check_same_dtype_and_shape(tensor, tensor_info, name):
    """Validate that tensor has the same properties as the TensorInfo proto.

  Args:
    tensor: a `Tensor` object.
    tensor_info: a `TensorInfo` proto.
    name: Name of the input (to identify Tensor if an error is raised).

  Raises:
    ValueError: If the tensor shape or dtype don't match the TensorInfo
  """
    dtype_error = tensor.dtype != tf.dtypes.DType(tensor_info.dtype)
    shape_error = not tensor.shape.is_compatible_with(tensor_info.tensor_shape)
    if dtype_error or shape_error:
        msg = 'Tensor shape and/or dtype validation failed for input %s:' % name
        if dtype_error:
            msg += '\n\tExpected dtype: %s, Got: %s' % (tf.dtypes.DType(tensor_info.dtype), tensor.dtype)
        if shape_error:
            msg += '\n\tExpected shape: %s, Got: %s' % (tf.TensorShape(tensor_info.tensor_shape), tensor.shape)
        raise ValueError(msg)