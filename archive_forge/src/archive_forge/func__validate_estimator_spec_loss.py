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
def _validate_estimator_spec_loss(loss, mode):
    """Validate loss inputs for EstimatorSpec or TPUEstimatorSpec.

  Args:
    loss: Training loss `Tensor`. Must either be scalar, or with shape `[1]`.
    mode: A `ModeKeys`. Used to determine whether the loss is acceptable for use
      in the current mode; for example, None is acceptable if we are not
      training or evaluating.

  Returns:
    loss: Training loss `Tensor`.

  Raises:
    ValueError: If the loss `Tensor` is not appropriately formatted.
    TypeError:  If:
                - a non-`Tensor`, non-None input is passed.
                - the loss `Tensor` is not part of the default graph.
  """
    if loss is None:
        if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
            raise ValueError('Missing loss.')
    else:
        default_graph = tf.compat.v1.get_default_graph()
        loss = _check_is_tensor(loss, 'loss')
        loss_shape = loss.get_shape()
        if loss_shape.num_elements() not in (None, 1):
            raise ValueError('Loss must be scalar, given: {}'.format(loss))
        if not loss_shape.is_compatible_with(tf.TensorShape([])):
            loss = tf.reshape(loss, [])
        if not (tf.executing_eagerly() or loss.graph is default_graph):
            raise ValueError(_default_graph_error_message_template.format('loss', loss.name))
    return loss