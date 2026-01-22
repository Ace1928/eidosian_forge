from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _linear_model_fn_v2(features, labels, mode, head, feature_columns, optimizer, config, sparse_combiner='sum'):
    """A model_fn for linear models that use a gradient-based optimizer.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape `[batch_size, logits_dimension]`.
    mode: Defines whether this is training, evaluation or prediction. See
      `ModeKeys`.
    head: A `Head` instance.
    feature_columns: An iterable containing all the feature columns used by the
      model.
    optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training. If `None`, will use a FTRL optimizer.
    config: `RunConfig` object to configure the runtime settings.
    sparse_combiner: A string specifying how to reduce if a categorical column
      is multivalent.  One of "mean", "sqrtn", and "sum".

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: mode or params are invalid, or features has the wrong type.
  """
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. Given type: {}'.format(type(features)))
    del config
    if isinstance(optimizer, LinearSDCA):
        assert sparse_combiner == 'sum'
        return _sdca_model_fn(features, labels, mode, head, feature_columns, optimizer)
    else:
        logits, trainable_variables = _linear_model_fn_builder_v2(units=head.logits_dimension, feature_columns=feature_columns, sparse_combiner=sparse_combiner, features=features)
        if mode == ModeKeys.TRAIN:
            optimizer = optimizers.get_optimizer_instance_v2(optimizer or _get_default_optimizer_v2(feature_columns), learning_rate=_LEARNING_RATE)
            optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
        return head.create_estimator_spec(features=features, mode=mode, labels=labels, optimizer=optimizer, trainable_variables=trainable_variables, logits=logits)