from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _dnn_model_fn_builder_v2(units, hidden_units, feature_columns, activation_fn, dropout, batch_norm, features, mode):
    """Function builder for dnn logits, trainable variables and update ops.

  Args:
    units: An int indicating the dimension of the logit layer.  In the MultiHead
      case, this should be the sum of all component Heads' logit dimensions.
    hidden_units: Iterable of integer number of hidden units per layer.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    activation_fn: Activation function applied to each layer.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    batch_norm: Whether to use batch normalization after each hidden layer.
    features: This is the first item returned from the `input_fn` passed to
      `train`, `evaluate`, and `predict`. This should be a single `Tensor` or
      `dict` of same.
    mode: Optional. Specifies if this training, evaluation or prediction. See
      `ModeKeys`.

  Returns:
    A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    A list of trainable variables.
    A list of update ops.

  Raises:
    ValueError: If units is not an int.
  """
    if not isinstance(units, six.integer_types):
        raise ValueError('units must be an int.  Given type: {}'.format(type(units)))
    dnn_model = _DNNModelV2(units, hidden_units, feature_columns, activation_fn, dropout, batch_norm, name='dnn')
    logits = dnn_model(features, mode)
    trainable_variables = dnn_model.trainable_variables
    update_ops = dnn_model.updates
    return (logits, trainable_variables, update_ops)