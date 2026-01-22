from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.saved_model import model_utils as export_utils
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _validate_estimator_spec_predictions(predictions, mode):
    """Validate predictions inputs for EstimatorSpec or TPUEstimatorSpec.

  Args:
    predictions: Predictions `Tensor` or dict of `Tensor`.
    mode: A `ModeKeys`. Used to determine whether the predictions are acceptable
      for use in the current mode; None is acceptable if we are not making
      predictions.

  Returns:
    predictions: Predictions `Tensor` or dict of `Tensor`.

  Raises:
    ValueError: If:
      - predictions is None and we are in predict mode.
      - predictions `Tensor` is not in default_graph or else it is a dict of
        `Tensor` where at least one is not in default_graph.
    TypeError:  If predictions is not a `Tensor` or dict of `Tensor`.
  """
    if predictions is None:
        if mode == ModeKeys.PREDICT:
            raise ValueError('Missing predictions.')
        predictions = {}
    else:
        default_graph = tf.compat.v1.get_default_graph()
        if isinstance(predictions, dict):
            predictions = {k: _check_is_tensor(v, 'predictions[{}]'.format(k)) for k, v in six.iteritems(predictions)}
            if not tf.executing_eagerly():
                for key, value in six.iteritems(predictions):
                    if value.graph is not default_graph:
                        raise ValueError(_default_graph_error_message_template.format('prediction values', '{0}: {1}'.format(key, value.name)))
        else:
            predictions = _check_is_tensor(predictions, 'predictions')
            if not (tf.executing_eagerly() or predictions.graph is default_graph):
                raise ValueError(_default_graph_error_message_template.format('prediction values', predictions.name))
    return predictions