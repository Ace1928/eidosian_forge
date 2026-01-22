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
@estimator_export('estimator.DNNLinearCombinedEstimator', v1=[])
class DNNLinearCombinedEstimatorV2(estimator.EstimatorV2):
    """An estimator for TensorFlow Linear and DNN joined models with custom head.

  Note: This estimator is also known as wide-n-deep.

  Example:

  ```python
  numeric_feature = numeric_column(...)
  categorical_column_a = categorical_column_with_hash_bucket(...)
  categorical_column_b = categorical_column_with_hash_bucket(...)

  categorical_feature_a_x_categorical_feature_b = crossed_column(...)
  categorical_feature_a_emb = embedding_column(
      categorical_column=categorical_feature_a, ...)
  categorical_feature_b_emb = embedding_column(
      categorical_column=categorical_feature_b, ...)

  estimator = tf.estimator.DNNLinearCombinedEstimator(
      head=tf.estimator.MultiLabelHead(n_classes=3),
      # wide settings
      linear_feature_columns=[categorical_feature_a_x_categorical_feature_b],
      linear_optimizer=tf.keras.optimizers.Ftrl(...),
      # deep settings
      dnn_feature_columns=[
          categorical_feature_a_emb, categorical_feature_b_emb,
          numeric_feature],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.keras.optimizers.Adagrad(...))

  # To apply L1 and L2 regularization, you can set dnn_optimizer to:
  tf.compat.v1.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001,
      l2_regularization_strength=0.001)
  # To apply learning rate decay, you can set dnn_optimizer to a callable:
  lambda: tf.keras.optimizers.Adam(
      learning_rate=tf.compat.v1.train.exponential_decay(
          learning_rate=0.1,
          global_step=tf.compat.v1.train.get_global_step(),
          decay_steps=10000,
          decay_rate=0.96)
  # It is the same for linear_optimizer.

  # Input builders
  def input_fn_train:
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    pass
  def input_fn_eval:
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    pass
  def input_fn_predict:
    # Returns tf.data.Dataset of (x, None) tuple.
    pass
  estimator.train(input_fn=input_fn_train, steps=100)
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
    - if `column` is a `CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss is calculated by using mean squared error.

  @compatibility(eager)
  Estimators can be used while eager execution is enabled. Note that `input_fn`
  and all hooks are executed inside a graph context, so they have to be written
  to be compatible with graph mode. Note that `input_fn` code using `tf.data`
  generally works in both graph and eager modes.
  @end_compatibility
  """

    def __init__(self, head, model_dir=None, linear_feature_columns=None, linear_optimizer='Ftrl', dnn_feature_columns=None, dnn_optimizer='Adagrad', dnn_hidden_units=None, dnn_activation_fn=tf.nn.relu, dnn_dropout=None, config=None, batch_norm=False, linear_sparse_combiner='sum'):
        """Initializes a DNNLinearCombinedEstimator instance.

    Args:
      head: A `Head` instance constructed with a method such as
        `tf.estimator.MultiLabelHead`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator to
        continue training a previously saved model.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be instances
        of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.keras.optimizers.*` used to apply
        gradients to the linear part of the model. Can also be a string (one of
        'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to
        FTRL optimizer.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set must be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.keras.optimizers.*` used to apply
        gradients to the deep part of the model. Can also be a string (one of
        'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to
        Adagrad optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If None,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out a given
        coordinate.
      config: RunConfig object to configure the runtime settings.
      batch_norm: Whether to use batch normalization after each hidden layer.
      linear_sparse_combiner: A string specifying how to reduce the linear model
        if a categorical column is multivalent.  One of "mean", "sqrtn", and
        "sum" -- these are effectively different ways to do example-level
        normalization, which can be useful for bag-of-words features.  For more
        details, see `tf.feature_column.linear_model`.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
        self._feature_columns = _validate_feature_columns(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns)
        estimator._canned_estimator_api_gauge.get_cell('Estimator').set('DNNLinearCombined')

        def _model_fn(features, labels, mode, config):
            """Call the _dnn_linear_combined_model_fn."""
            return _dnn_linear_combined_model_fn_v2(features=features, labels=labels, mode=mode, head=head, linear_feature_columns=linear_feature_columns, linear_optimizer=linear_optimizer, dnn_feature_columns=dnn_feature_columns, dnn_optimizer=dnn_optimizer, dnn_hidden_units=dnn_hidden_units, dnn_activation_fn=dnn_activation_fn, dnn_dropout=dnn_dropout, config=config, batch_norm=batch_norm, linear_sparse_combiner=linear_sparse_combiner)
        super(DNNLinearCombinedEstimatorV2, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)