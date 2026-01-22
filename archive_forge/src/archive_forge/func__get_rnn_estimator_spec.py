from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head as binary_head_lib
from tensorflow_estimator.python.estimator.head import multi_class_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import sequential_head as seq_head_lib
def _get_rnn_estimator_spec(features, labels, mode, head, rnn_model, optimizer, return_sequences):
    """Computes `EstimatorSpec` from logits to use in estimator model function.

  Args:
    features: dict of `Tensor` and `SparseTensor` objects returned from
      `input_fn`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] with labels.
    mode: Defines whether this is training, evaluation or prediction. See
      `ModeKeys`.
    head: A `Head` instance.
    rnn_model: A Keras model that computes RNN logits from features.
    optimizer: String, `tf.keras.optimizers.Optimizer` object, or callable that
      creates the optimizer to use for training. If not specified, will use the
      Adagrad optimizer with a default learning rate of 0.05 and gradient clip
      norm of 5.0.
    return_sequences: A boolean indicating whether to return the last output in
      the output sequence, or the full sequence.

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If mode or optimizer is invalid, or features has the wrong type.
  """
    training = mode == model_fn.ModeKeys.TRAIN
    if training:
        if isinstance(optimizer, six.string_types):
            optimizer = optimizers.get_optimizer_instance_v2(optimizer, learning_rate=_DEFAULT_LEARNING_RATE)
            optimizer.clipnorm = _DEFAULT_CLIP_NORM
        else:
            optimizer = optimizers.get_optimizer_instance_v2(optimizer)
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    else:
        optimizer = None
    logits = rnn_model(features, training)
    if return_sequences and head.input_sequence_mask_key not in features:
        features[head.input_sequence_mask_key] = logits._keras_mask
    return head.create_estimator_spec(features=features, mode=mode, labels=labels, optimizer=optimizer, logits=logits, update_ops=rnn_model.updates, trainable_variables=rnn_model.trainable_variables)