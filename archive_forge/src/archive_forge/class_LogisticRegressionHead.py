from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export('estimator.LogisticRegressionHead')
class LogisticRegressionHead(RegressionHead):
    """Creates a `Head` for logistic regression.

  Uses `sigmoid_cross_entropy_with_logits` loss, which is the same as
  `BinaryClassHead`. The differences compared to `BinaryClassHead` are:

  * Does not support `label_vocabulary`. Instead, labels must be float in the
    range [0, 1].
  * Does not calculate some metrics that do not make sense, such as AUC.
  * In `PREDICT` mode, only returns logits and predictions
    (`=tf.sigmoid(logits)`), whereas `BinaryClassHead` also returns
    probabilities, classes, and class_ids.
  * Export output defaults to `RegressionOutput`, whereas `BinaryClassHead`
    defaults to `PredictOutput`.

  The head expects `logits` with shape `[D0, D1, ... DN, 1]`.
  In many applications, the shape is `[batch_size, 1]`.

  The `labels` shape must match `logits`, namely
  `[D0, D1, ... DN]` or `[D0, D1, ... DN, 1]`.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]` or `[D0, D1, ... DN, 1]`.

  This is implemented as a generalized linear model, see
  https://en.wikipedia.org/wiki/Generalized_linear_model.

  The head can be used with a canned estimator. Example:

  ```python
  my_head = tf.estimator.LogisticRegressionHead()
  my_estimator = tf.estimator.DNNEstimator(
      head=my_head,
      hidden_units=...,
      feature_columns=...)
  ```

  It can also be used with a custom `model_fn`. Example:

  ```python
  def _my_model_fn(features, labels, mode):
    my_head = tf.estimator.LogisticRegressionHead()
    logits = tf.keras.Model(...)(features)

    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.keras.optimizers.Adagrad(lr=0.1),
        logits=logits)

  my_estimator = tf.estimator.Estimator(model_fn=_my_model_fn)
  ```

  Args:
    weight_column: A string or a `NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Decides how to
      reduce training loss over batch and label dimension. Defaults to
      `SUM_OVER_BATCH_SIZE`, namely weighted sum of losses divided by `batch
      size * label_dimension`.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.
  """

    def _logistic_loss(self, labels, logits):
        labels = base_head.check_label_range(labels, n_classes=2, message='Labels must be in range [0, 1]')
        return tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    def __init__(self, weight_column=None, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE, name=None):
        super(LogisticRegressionHead, self).__init__(label_dimension=1, weight_column=weight_column, loss_reduction=loss_reduction, loss_fn=self._logistic_loss, inverse_link_fn=tf.math.sigmoid, name=name)