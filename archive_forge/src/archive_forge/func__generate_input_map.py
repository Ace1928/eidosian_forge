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
def _generate_input_map(signature_def, features, labels):
    """Return dict mapping an input tensor name to a feature or label tensor.

  Args:
    signature_def: SignatureDef loaded from SavedModel
    features: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the features to be passed to the model.
    labels: A `Tensor`, `SparseTensor`, or dict of string to `Tensor` or
      `SparseTensor`, specifying the labels to be passed to the model. May be
      `None`.

  Returns:
    dict mapping string names of inputs to features or labels tensors

  Raises:
    ValueError: if SignatureDef inputs are not completely mapped by the input
      features and labels.
  """
    features = export_lib.wrap_and_check_input_tensors(features, 'feature')
    if labels is not None:
        labels = export_lib.wrap_and_check_input_tensors(labels, 'label')
    inputs = signature_def.inputs
    input_map = {}
    for key, tensor_info in six.iteritems(inputs):
        input_name = tensor_info.name
        if ':' in input_name:
            input_name = input_name[:input_name.find(':')]
        control_dependency_name = '^' + input_name
        if key in features:
            _check_same_dtype_and_shape(features[key], tensor_info, key)
            input_map[input_name] = input_map[control_dependency_name] = features[key]
        elif labels is not None and key in labels:
            _check_same_dtype_and_shape(labels[key], tensor_info, key)
            input_map[input_name] = input_map[control_dependency_name] = labels[key]
        else:
            raise ValueError('Key "%s" not found in features or labels passed in to the model function. All required keys: %s' % (key, inputs.keys()))
    return input_map