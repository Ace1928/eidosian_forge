from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _dnn_linear_combined_model_fn_v2(features, labels, mode, head, linear_feature_columns=None, linear_optimizer='Ftrl', dnn_feature_columns=None, dnn_optimizer='Adagrad', dnn_hidden_units=None, dnn_activation_fn=tf.nn.relu, dnn_dropout=None, config=None, batch_norm=False, linear_sparse_combiner='sum', loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE):
    """Deep Neural Net and Linear combined model_fn.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
      `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction. See
      `ModeKeys`.
    head: A `Head` instance.
    linear_feature_columns: An iterable containing all the feature columns used
      by the Linear model.
    linear_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the Linear model. Defaults to the Ftrl
      optimizer.
    dnn_feature_columns: An iterable containing all the feature columns used by
      the DNN model.
    dnn_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the DNN model. Defaults to the Adagrad
      optimizer.
    dnn_hidden_units: List of hidden units per DNN layer.
    dnn_activation_fn: Activation function applied to each DNN layer. If `None`,
      will use `tf.nn.relu`.
    dnn_dropout: When not `None`, the probability we will drop out a given DNN
      coordinate.
    config: `RunConfig` object to configure the runtime settings.
    batch_norm: Whether to use batch normalization after each hidden layer.
    linear_sparse_combiner: A string specifying how to reduce the linear model
      if a categorical column is multivalent.  One of "mean", "sqrtn", and
      "sum".
    loss_reduction: One of `tf.keras.losses.Reduction` except `NONE`. Describes
      how to reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If both `linear_feature_columns` and `dnn_features_columns`
      are empty at the same time, or `input_layer_partitioner` is missing,
      or features has the wrong type.
  """
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. Given type: {}'.format(type(features)))
    if not linear_feature_columns and (not dnn_feature_columns):
        raise ValueError('Either linear_feature_columns or dnn_feature_columns must be defined.')
    del config
    if not dnn_feature_columns:
        dnn_logits = None
    else:
        if mode == ModeKeys.TRAIN:
            dnn_optimizer = optimizers.get_optimizer_instance_v2(dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
            _check_no_sync_replicas_optimizer(dnn_optimizer)
        if not dnn_hidden_units:
            raise ValueError('dnn_hidden_units must be defined when dnn_feature_columns is specified.')
        dnn_logits, dnn_trainable_variables, dnn_update_ops = dnn._dnn_model_fn_builder_v2(units=head.logits_dimension, hidden_units=dnn_hidden_units, feature_columns=dnn_feature_columns, activation_fn=dnn_activation_fn, dropout=dnn_dropout, batch_norm=batch_norm, features=features, mode=mode)
    if not linear_feature_columns:
        linear_logits = None
    else:
        if mode == ModeKeys.TRAIN:
            linear_optimizer = optimizers.get_optimizer_instance_v2(linear_optimizer, learning_rate=_linear_learning_rate(len(linear_feature_columns)))
            _check_no_sync_replicas_optimizer(linear_optimizer)
        linear_logits, linear_trainable_variables = linear._linear_model_fn_builder_v2(units=head.logits_dimension, feature_columns=linear_feature_columns, sparse_combiner=linear_sparse_combiner, features=features)
        _add_layer_summary(linear_logits, 'linear')
    if dnn_logits is not None and linear_logits is not None:
        logits = dnn_logits + linear_logits
    elif dnn_logits is not None:
        logits = dnn_logits
    else:
        logits = linear_logits

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        train_ops = []
        if loss_reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE:
            num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
            if num_replicas > 1:
                loss *= 1.0 / num_replicas
        if dnn_logits is not None:
            train_ops.extend(dnn_optimizer.get_updates(loss, dnn_trainable_variables))
            if dnn_update_ops is not None:
                train_ops.extend(dnn_update_ops)
        if linear_logits is not None:
            train_ops.extend(linear_optimizer.get_updates(loss, linear_trainable_variables))
        train_op = tf.group(*train_ops)
        return train_op
    if mode == ModeKeys.TRAIN:
        if dnn_logits is not None:
            dnn_optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
        else:
            linear_optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    return head.create_estimator_spec(features=features, mode=mode, labels=labels, train_op_fn=_train_op_fn, logits=logits)